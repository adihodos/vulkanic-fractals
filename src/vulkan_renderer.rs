use std::ffi::CString;

use ash::vk::{
    AccessFlags2, AttachmentLoadOp, AttachmentStoreOp, CommandBuffer, CommandBufferSubmitInfo,
    DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT, DependencyFlags,
    DependencyInfo, DescriptorSetLayout, DeviceMemory, Extent2D, Extent3D, Fence, Format,
    Framebuffer, Handle, Image, ImageAspectFlags, ImageCreateInfo, ImageLayout,
    ImageMemoryBarrier2, ImageSubresourceRange, ImageTiling, ImageType, ImageUsageFlags, ImageView,
    ImageViewCreateInfo, ImageViewType, MemoryAllocateInfo, MemoryPropertyFlags, Offset2D,
    PipelineBindPoint, PipelineStageFlags2, Rect2D, SampleCountFlags, Semaphore,
    SemaphoreSubmitInfo, SharingMode,
};

use smallvec::SmallVec;

use crate::{spin_mutex::SpinMutex, vulkan_bindless::BindlessImageResourceHandleEntryPair};

pub trait VkObjectType {
    fn object_type() -> ash::vk::ObjectType;
}

macro_rules! debug_object_tag_helper {
    ($vkobj:ty, $tag:expr) => {
        impl VkObjectType for $vkobj {
            fn object_type() -> ash::vk::ObjectType {
                $tag
            }
        }
    };
}

debug_object_tag_helper!(ash::vk::Buffer, ash::vk::ObjectType::BUFFER);
debug_object_tag_helper!(ash::vk::Image, ash::vk::ObjectType::IMAGE);
debug_object_tag_helper!(ash::vk::ImageView, ash::vk::ObjectType::IMAGE_VIEW);
debug_object_tag_helper!(ash::vk::CommandBuffer, ash::vk::ObjectType::COMMAND_BUFFER);
debug_object_tag_helper!(ash::vk::Pipeline, ash::vk::ObjectType::PIPELINE);
debug_object_tag_helper!(
    ash::vk::PipelineLayout,
    ash::vk::ObjectType::PIPELINE_LAYOUT
);
debug_object_tag_helper!(ash::vk::DescriptorSet, ash::vk::ObjectType::DESCRIPTOR_SET);
debug_object_tag_helper!(
    ash::vk::DescriptorSetLayout,
    ash::vk::ObjectType::DESCRIPTOR_SET_LAYOUT
);
debug_object_tag_helper!(ash::vk::DeviceMemory, ash::vk::ObjectType::DEVICE_MEMORY);

#[derive(thiserror::Error, Debug)]
pub enum GraphicsError {
    #[error("I/O error, file {f}, error {e}")]
    IoError {
        f: std::path::PathBuf,
        e: std::io::Error,
    },
    #[error("shaderc generic error: {0}")]
    ShadercGeneric(String),
    #[error("shaderc compile error")]
    ShadercCompilationError(#[from] shaderc::Error),
    #[error("Vulkan API error")]
    VulkanApi(#[from] ash::vk::Result),
    #[error("SPIRV reflection error")]
    SpirVReflectionError(&'static str),
    #[error("Other graphics error")]
    Generic(String),
}

pub struct VulkanSurfaceWithInstance {
    pub ext: ash::khr::surface::Instance,
    pub surface: ash::vk::SurfaceKHR,
}

impl std::ops::Drop for VulkanSurfaceWithInstance {
    fn drop(&mut self) {
        unsafe {
            self.ext.destroy_surface(self.surface, None);
        }
    }
}

pub struct VulkanDeviceState {
    debug: ash::ext::debug_utils::Device,
    pipeline_cache: ash::vk::PipelineCache,
    null_descriptor_layout: ash::vk::DescriptorSetLayout,
    pub render_state: RenderState,
    pub physical: PhysicalDeviceState,
    pub logical: ash::Device,
}

impl std::ops::Drop for VulkanDeviceState {
    fn drop(&mut self) {
        unsafe {
            self.logical
                .destroy_pipeline_cache(self.pipeline_cache, None);
            self.logical
                .destroy_descriptor_set_layout(self.null_descriptor_layout, None);
            self.logical.destroy_device(None);
        }
    }
}

impl VulkanDeviceState {
    pub fn create(
        instance: &ash::Instance,
        physical: PhysicalDeviceState,
        surface: &VulkanSurfaceState,
        queue_families: &[u32],
        queue_priorities: &[f32],
    ) -> std::result::Result<VulkanDeviceState, GraphicsError> {
        let mut feature_shader_draw_params =
            ash::vk::PhysicalDeviceShaderDrawParametersFeatures::default()
                .shader_draw_parameters(true);

        let mut feature_descriptor_indexing =
            ash::vk::PhysicalDeviceDescriptorIndexingFeatures::default()
                .shader_input_attachment_array_dynamic_indexing(true)
                .shader_uniform_texel_buffer_array_dynamic_indexing(true)
                .shader_storage_texel_buffer_array_dynamic_indexing(true)
                .shader_uniform_buffer_array_non_uniform_indexing(true)
                .shader_sampled_image_array_non_uniform_indexing(true)
                .shader_storage_buffer_array_non_uniform_indexing(true)
                .shader_storage_image_array_non_uniform_indexing(true)
                .shader_input_attachment_array_non_uniform_indexing(true)
                .shader_uniform_texel_buffer_array_non_uniform_indexing(true)
                .shader_storage_texel_buffer_array_non_uniform_indexing(true)
                .descriptor_binding_sampled_image_update_after_bind(true)
                .descriptor_binding_storage_image_update_after_bind(true)
                .descriptor_binding_storage_buffer_update_after_bind(true)
                .descriptor_binding_uniform_texel_buffer_update_after_bind(true)
                .descriptor_binding_storage_texel_buffer_update_after_bind(true)
                .descriptor_binding_update_unused_while_pending(true)
                .descriptor_binding_partially_bound(true)
                .descriptor_binding_variable_descriptor_count(true)
                .runtime_descriptor_array(true);

        let mut feature_dynamic_rendering =
            ash::vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

        let mut feature_sync2 =
            ash::vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);

        let mut enabled_features = ash::vk::PhysicalDeviceFeatures2::default()
            .features(physical.features.base)
            .push_next(&mut feature_shader_draw_params)
            .push_next(&mut feature_descriptor_indexing)
            .push_next(&mut feature_dynamic_rendering)
            .push_next(&mut feature_sync2);

        let queue_create_infos = queue_families
            .iter()
            .enumerate()
            .map(|(idx, family_id)| {
                ash::vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(*family_id)
                    .queue_priorities(&queue_priorities[idx..idx + 1])
            })
            .collect::<SmallVec<[ash::vk::DeviceQueueCreateInfo; 4]>>();

        let logical = unsafe {
            instance.create_device(
                physical.device,
                &ash::vk::DeviceCreateInfo::default()
                    .push_next(&mut enabled_features)
                    .enabled_extension_names(&[
                        ash::vk::KHR_SWAPCHAIN_NAME.as_ptr(),
                        ash::vk::EXT_DESCRIPTOR_BUFFER_NAME.as_ptr(),
                    ])
                    .queue_create_infos(&queue_create_infos),
                None,
            )
        }?;

        let render_state = Self::create_render_state(&physical, &logical, surface)?;
        let pipeline_cache = unsafe {
            logical.create_pipeline_cache(&ash::vk::PipelineCacheCreateInfo::default(), None)
        }?;
        let null_descriptor_layout = unsafe {
            logical.create_descriptor_set_layout(
                &ash::vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(ash::vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL),
                None,
            )
        }?;

        Ok(VulkanDeviceState {
            debug: ash::ext::debug_utils::Device::new(instance, &logical),
            physical,
            logical,
            render_state,
            pipeline_cache,
            null_descriptor_layout,
        })
    }

    fn create_render_state(
        physical: &PhysicalDeviceState,
        logical: &ash::Device,
        surface: &VulkanSurfaceState,
    ) -> std::result::Result<RenderState, GraphicsError> {
        if physical.features.dynamic_rendering {
            Ok(RenderState::Dynamic {
                color_attachments: [surface.format.format],
                depth_attachments: [surface.format.format],
                stencil_attachments: [surface.format.format],
            })
        } else {
            let attachment_descriptions = [ash::vk::AttachmentDescription::default()
                .format(surface.format.format)
                .samples(SampleCountFlags::TYPE_1)
                .load_op(AttachmentLoadOp::CLEAR)
                .store_op(AttachmentStoreOp::STORE)
                .stencil_load_op(AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(AttachmentStoreOp::DONT_CARE)
                .initial_layout(ImageLayout::UNDEFINED)
                .final_layout(ImageLayout::PRESENT_SRC_KHR)];

            let attachment_refs = [ash::vk::AttachmentReference::default()
                .attachment(0)
                .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

            unsafe {
                logical.create_render_pass(
                    &ash::vk::RenderPassCreateInfo::default()
                        .attachments(&attachment_descriptions)
                        .subpasses(&[ash::vk::SubpassDescription::default()
                            .color_attachments(&attachment_refs)
                            .pipeline_bind_point(PipelineBindPoint::GRAPHICS)]),
                    None,
                )
            }
            .map_err(|e| GraphicsError::from(e))
            .map(|vk_pass| RenderState::Renderpass(vk_pass))
        }
    }
}

struct QueueState {
    family_indices: Vec<u32>,
    handles: Vec<ash::vk::Queue>,
    cmd_pools: Vec<ash::vk::CommandPool>,
    pool_locks: Vec<SpinMutex>,
    queue_locks: Vec<SpinMutex>,
}

impl QueueState {
    fn destroy(&mut self, device: &ash::Device) {
        std::mem::take(&mut self.cmd_pools)
            .into_iter()
            .for_each(|cmd_pool| unsafe {
                device.destroy_command_pool(cmd_pool, None);
            });
    }

    fn create(
        device: &ash::Device,
        family_indices: &[u32],
    ) -> std::result::Result<QueueState, GraphicsError> {
        let cmd_pools: Result<Vec<ash::vk::CommandPool>, GraphicsError> = family_indices
            .iter()
            .map(|&family_index| unsafe {
                device
                    .create_command_pool(
                        &ash::vk::CommandPoolCreateInfo::default()
                            .queue_family_index(family_index)
                            .flags(
                                ash::vk::CommandPoolCreateFlags::TRANSIENT
                                    | ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                            ),
                        None,
                    )
                    .map_err(|e| GraphicsError::from(e))
            })
            .collect();

        Ok(QueueState {
            family_indices: Vec::from(family_indices),
            handles: family_indices
                .iter()
                .map(|family_index| unsafe { device.get_device_queue(*family_index, 0) })
                .collect(),
            cmd_pools: cmd_pools?,
            pool_locks: vec![SpinMutex::new(), SpinMutex::new()],
            queue_locks: vec![SpinMutex::new(), SpinMutex::new()],
        })
    }
}

pub struct VulkanSurfaceState {
    pub surface: VulkanSurfaceWithInstance,
    pub caps: ash::vk::SurfaceCapabilitiesKHR,
    pub format: ash::vk::SurfaceFormatKHR,
    pub depth_format: ash::vk::Format,
    pub present_mode: ash::vk::PresentModeKHR,
}

pub struct PhysicalDeviceProperties {
    base: ash::vk::PhysicalDeviceProperties,
}

pub struct PhysicalDeviceFeatures {
    base: ash::vk::PhysicalDeviceFeatures,
    dynamic_rendering: bool,
}

pub struct PhysicalDeviceState {
    device: ash::vk::PhysicalDevice,
    properties: PhysicalDeviceProperties,
    features: PhysicalDeviceFeatures,
    memory: ash::vk::PhysicalDeviceMemoryProperties,
}

pub struct InstanceState {
    debug_utils_msgr: ash::vk::DebugUtilsMessengerEXT,
    debug_utils_inst: ash::ext::debug_utils::Instance,
    handle: ash::Instance,
    entry: ash::Entry,
}

impl std::ops::Drop for InstanceState {
    fn drop(&mut self) {
        unsafe {
            self.debug_utils_inst
                .destroy_debug_utils_messenger(std::mem::take(&mut self.debug_utils_msgr), None);
            self.handle.destroy_instance(None);
        }
    }
}

impl InstanceState {
    pub fn create() -> std::result::Result<InstanceState, GraphicsError> {
        let entry = ash::Entry::linked();
        let instance_ver = unsafe { entry.try_enumerate_instance_version() }?;

        log::info!(
            "Vulkan version: {}.{}.{}",
            instance_ver.map_or(1, |instance_ver| {
                ash::vk::api_version_major(instance_ver)
            },),
            instance_ver.map_or(0, |instance_ver| {
                ash::vk::api_version_minor(instance_ver)
            },),
            instance_ver.map_or(0, |instance_ver| {
                ash::vk::api_version_patch(instance_ver)
            },)
        );

        unsafe {
            log::info!("Instance extensions:");
            let _ = entry
                .enumerate_instance_extension_properties(None)
                .map(|instance_exts| {
                    instance_exts.iter().for_each(|ext| {
                        let ext_name = std::ffi::CStr::from_ptr(ext.extension_name.as_ptr())
                            .to_str()
                            .unwrap();
                        log::info!("{ext_name}");
                    });
                });
        }

        let mut debug_utils_msg_create_info = ash::vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | DebugUtilsMessageSeverityFlagsEXT::INFO
                    | DebugUtilsMessageSeverityFlagsEXT::WARNING,
            )
            .message_type(
                DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(debug_callback_stub));

        let app_name_engine = [
            CString::new("Vulkan + Rust adventures").unwrap(),
            CString::new("Epic engine").unwrap(),
        ];

        let app_create_info = ash::vk::ApplicationInfo::default()
            .api_version(ash::vk::make_api_version(0, 1, 3, 0))
            .application_name(app_name_engine[0].as_c_str())
            .application_version(1)
            .engine_name(app_name_engine[1].as_c_str())
            .engine_version(1);

        let enabled_layers = [b"VK_LAYER_KHRONOS_validation\0".as_ptr() as *const std::ffi::c_char];

        let enabled_instance_extensions = [
            ash::vk::KHR_SURFACE_NAME.as_ptr(),
            #[cfg(target_os = "linux")]
            ash::vk::KHR_XLIB_SURFACE_NAME.as_ptr(),
            #[cfg(target_os = "windows")]
            ash::vk::KHR_WIN32_SURFACE_NAME.as_ptr(),
            ash::vk::EXT_DEBUG_UTILS_NAME.as_ptr(),
        ];

        let instance = unsafe {
            entry.create_instance(
                &ash::vk::InstanceCreateInfo::default()
                    .application_info(&app_create_info)
                    .push_next(&mut debug_utils_msg_create_info)
                    .enabled_layer_names(&enabled_layers)
                    .enabled_extension_names(&enabled_instance_extensions),
                None,
            )
        }
        .expect("Failed to create instance");

        let (debug_utils_inst, debug_utils_msgr) = unsafe {
            let debug_utils_inst = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let debug_utils_msgr = debug_utils_inst
                .create_debug_utils_messenger(&debug_utils_msg_create_info, None)?;

            (debug_utils_inst, debug_utils_msgr)
        };

        Ok(InstanceState {
            debug_utils_msgr,
            debug_utils_inst,
            handle: instance,
            entry,
        })
    }
}

unsafe extern "system" fn debug_callback_stub(
    message_severity: ash::vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_types: ash::vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const ash::vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::os::raw::c_void,
) -> ash::vk::Bool32 {
    if message_severity.intersects(DebugUtilsMessageSeverityFlagsEXT::INFO) {
        log::info!(
            "{}",
            std::ffi::CStr::from_ptr((*p_callback_data).p_message)
                .to_str()
                .unwrap_or_else(|_| "unknown")
        );
    }

    if message_severity.intersects(DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        log::warn!(
            "{}",
            std::ffi::CStr::from_ptr((*p_callback_data).p_message)
                .to_str()
                .unwrap_or_else(|_| "unknown")
        );
    }

    if message_severity.intersects(DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!(
            "{}",
            std::ffi::CStr::from_ptr((*p_callback_data).p_message)
                .to_str()
                .unwrap_or_else(|_| "unknown")
        );
    }

    ash::vk::FALSE
}

fn debug_set_object_name<T: VkObjectType + ash::vk::Handle>(
    debug: &ash::ext::debug_utils::Device,
    vkobject: T,
    name: &str,
) {
    let _ = unsafe {
        debug.set_debug_utils_object_name(
            &ash::vk::DebugUtilsObjectNameInfoEXT::default()
                .object_handle(vkobject)
                .object_name(&std::ffi::CString::new(name).unwrap()),
        )
    };
}

pub struct VulkanRenderer {
    staging_system: StagingSystem,
    presentation_state: PresentationState,
    queue_state: QueueState,
    device_state: VulkanDeviceState,
    surface_state: VulkanSurfaceState,
    _instance: InstanceState,
}

unsafe impl Send for VulkanRenderer {}
unsafe impl Sync for VulkanRenderer {}

impl std::ops::Drop for VulkanRenderer {
    fn drop(&mut self) {
        self.staging_system.destroy(&self.device_state.logical);
        self.presentation_state.destroy(&self.device_state.logical);
        self.queue_state.destroy(&self.device_state.logical);
    }
}

impl VulkanRenderer {
    pub fn create(
        wsi: WindowSystemIntegration,
    ) -> std::result::Result<VulkanRenderer, GraphicsError> {
        let instance = InstanceState::create()?;
        let surface_state = create_surface(&instance.entry, &instance.handle, wsi)?;

        let (phys_device_state, extra_data) =
            pick_device(&instance.handle, &surface_state.ext, surface_state.surface)?;

        let surface_state = VulkanSurfaceState {
            surface: surface_state,
            caps: extra_data.surface_caps,
            format: extra_data.surface_format,
            present_mode: extra_data.presentation_mode,
            depth_format: extra_data.depth_stencil_format,
        };

        let device_state = VulkanDeviceState::create(
            &instance.handle,
            phys_device_state,
            &surface_state,
            &[
                extra_data.queue_family_graphics,
                extra_data.queue_family_transfer,
            ],
            &[1.0, 1.0],
        )?;
        let queue_state = QueueState::create(
            &device_state.logical,
            &[
                extra_data.queue_family_graphics,
                extra_data.queue_family_transfer,
            ],
        )?;

        let swapchain_image_count = (((extra_data.surface_caps.min_image_count as f32) * 1.5f32)
            as u32)
            .min(extra_data.surface_caps.max_image_count);
        let swapchain_ext =
            ash::khr::swapchain::Device::new(&instance.handle, &device_state.logical);

        log::info!("Creating swapchain, surface caps: {:?}", surface_state.caps);
        let swapchain_state = VulkanSwapchainState::create(&VulkanSwapchainCreateInfo {
            retired_swapchain: None,
            surface: surface_state.surface.surface,
            image_count: swapchain_image_count,
            image_format: surface_state.format,
            depth_format: surface_state.depth_format,
            extent: surface_state.caps.current_extent,
            transform: surface_state.caps.current_transform,
            present_mode: surface_state.present_mode,
            ext: &swapchain_ext,
            device: &device_state.logical,
            memory_properties: &device_state.physical.memory,
            renderstate: &device_state.render_state,
        })?;

        let presentation_state = PresentationState::create(
            &device_state.logical,
            swapchain_ext,
            swapchain_state,
            swapchain_image_count,
        )?;

        let staging_system =
            StagingSystem::create(&device_state.logical, &device_state.physical.memory)?;

        Ok(VulkanRenderer {
            _instance: instance,
            surface_state,
            device_state,
            queue_state,
            presentation_state,
            staging_system,
        })
    }

    pub fn logical(&self) -> &ash::Device {
        &self.device_state.logical
    }

    pub fn logical_raw(&self) -> *const ash::Device {
        &self.device_state.logical as *const _
    }

    pub fn physical(&self) -> ash::vk::PhysicalDevice {
        self.device_state.physical.device
    }

    pub fn queue_data(
        &self,
        q: QueueType,
    ) -> (
        u32,
        ash::vk::Queue,
        ash::vk::CommandPool,
        &SpinMutex,
        &SpinMutex,
    ) {
        (
            self.queue_state.family_indices[q as usize],
            self.queue_state.handles[q as usize],
            self.queue_state.cmd_pools[q as usize],
            &self.queue_state.pool_locks[q as usize],
            &self.queue_state.queue_locks[q as usize],
        )
    }

    pub fn consume_wait_token(&mut self, token: QueueSubmitWaitToken) {
        unsafe {
            let _ = self
                .logical()
                .wait_for_fences(&[token.wait_fence], true, std::u64::MAX);
            self.logical().destroy_fence(token.wait_fence, None);
        }

        let (_, _, cmd_pool, pool_lock, _) = self.queue_data(token.queue_type);
        let _ = pool_lock.lock();
        unsafe {
            self.logical()
                .free_command_buffers(cmd_pool, &[token.cmd_buffer]);
        }
    }

    pub fn choose_memory_heap(
        &self,
        memory_req: &ash::vk::MemoryRequirements,
        memory_properties: MemoryPropertyFlags,
    ) -> u32 {
        choose_memory_heap(
            memory_req,
            memory_properties,
            &self.device_state.physical.memory,
        )
    }

    pub fn create_queue_job(&self, qtype: QueueType) -> Result<QueuedJob, GraphicsError> {
        let (_, _, cmd_pool, pool_lock, _) = self.queue_data(qtype);
        let command_buffer = {
            let _ = pool_lock.lock();
            unsafe {
                let cmd_buffers = self.logical().allocate_command_buffers(
                    &ash::vk::CommandBufferAllocateInfo::default()
                        .level(ash::vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1)
                        .command_pool(cmd_pool),
                )?;
                cmd_buffers[0]
            }
        };

        scopeguard::defer_on_unwind! {
            let _ = pool_lock.lock();
            unsafe {self.logical().free_command_buffers(cmd_pool, &[command_buffer]); }
        }

        unsafe {
            self.logical().begin_command_buffer(
                command_buffer,
                &ash::vk::CommandBufferBeginInfo::default()
                    .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        Ok(QueuedJob {
            cmd_buffer: command_buffer,
            queue_type: qtype,
        })
    }

    pub fn submit_queue_job(&self, job: QueuedJob) -> Result<QueueSubmitWaitToken, GraphicsError> {
        let (_, queue, cmd_pool, pool_lock, queue_lock) = self.queue_data(job.queue_type);

        let wait_fence = unsafe {
            self.device_state
                .logical
                .end_command_buffer(job.cmd_buffer)?;
            self.device_state
                .logical
                .create_fence(&ash::vk::FenceCreateInfo::default(), None)
        }?;

        scopeguard::defer_on_unwind! {
            unsafe {
                self.device_state.logical.destroy_fence(wait_fence, None);
                let _ = pool_lock.lock();
                self.device_state.logical.free_command_buffers(cmd_pool, &[job.cmd_buffer]);
            }
        }

        unsafe {
            self.device_state.logical.queue_submit(
                queue,
                &[ash::vk::SubmitInfo::default().command_buffers(&[job.cmd_buffer])],
                wait_fence,
            )?;
        }

        Ok(QueueSubmitWaitToken {
            cmd_buffer: job.cmd_buffer,
            wait_fence,
            queue_type: job.queue_type,
        })
    }

    pub fn reserve_staging_memory(&self, bytes: usize) -> (*mut u8, ash::vk::Buffer, usize) {
        let byte_offset = self
            .staging_system
            .free_ptr
            .fetch_add(bytes, std::sync::atomic::Ordering::Acquire);

        (
            unsafe {
                (self.staging_system.mapped_memory as *mut u8).byte_offset(byte_offset as isize)
            },
            self.staging_system.staging_buffer,
            byte_offset,
        )
    }

    pub fn limits(&self) -> &ash::vk::PhysicalDeviceLimits {
        &self.device_state.physical.properties.base.limits
    }

    pub fn memory_properties(&self) -> &ash::vk::PhysicalDeviceMemoryProperties {
        &self.device_state.physical.memory
    }

    pub fn features(&self) -> &ash::vk::PhysicalDeviceFeatures {
        &self.device_state.physical.features.base
    }

    pub fn empty_descriptor_set_layout(&self) -> DescriptorSetLayout {
        self.device_state.null_descriptor_layout
    }

    pub fn render_state(&self) -> RenderState {
        self.device_state.render_state
    }

    pub fn debug_set_object_name<T: VkObjectType + ash::vk::Handle>(
        &self,
        vkobject: T,
        name: &str,
    ) {
        debug_set_object_name(&self.device_state.debug, vkobject, name)
    }

    pub fn debug_marker_begin(&self, cmd_buf: CommandBuffer, name: &str, color: [f32; 4]) {
        let _ = unsafe {
            self.device_state.debug.cmd_begin_debug_utils_label(
                cmd_buf,
                &ash::vk::DebugUtilsLabelEXT::default()
                    .label_name(&std::ffi::CString::new(name).unwrap())
                    .color(color),
            )
        };
    }

    pub fn debug_insert_label(&self, cmd_buf: CommandBuffer, name: &str, color: [f32; 4]) {
        let _ = unsafe {
            self.device_state.debug.cmd_insert_debug_utils_label(
                cmd_buf,
                &ash::vk::DebugUtilsLabelEXT::default()
                    .label_name(&std::ffi::CString::new(name).unwrap())
                    .color(color),
            )
        };
    }

    pub fn debug_marker_end(&self, cmd_buf: CommandBuffer) {
        let _ = unsafe { self.device_state.debug.cmd_end_debug_utils_label(cmd_buf) };
    }

    pub fn debug_queue_begin_label(&self, name: &str, color: [f32; 4]) {
        let (_, queue, _, _, _) = self.queue_data(QueueType::Graphics);

        let _ = unsafe {
            self.device_state.debug.queue_begin_debug_utils_label(
                queue,
                &ash::vk::DebugUtilsLabelEXT::default()
                    .label_name(&std::ffi::CString::new(name).unwrap())
                    .color(color),
            )
        };
    }

    pub fn debug_queue_end_label(&self) {
        let (_, queue, _, _, _) = self.queue_data(QueueType::Graphics);
        let _ = unsafe { self.device_state.debug.queue_end_debug_utils_label(queue) };
    }

    pub fn debug_queue_insert_label(&self, queue: ash::vk::Queue, name: &str, color: [f32; 4]) {
        let _ = unsafe {
            self.device_state.debug.queue_insert_debug_utils_label(
                queue,
                &ash::vk::DebugUtilsLabelEXT::default()
                    .label_name(&std::ffi::CString::new(name).unwrap())
                    .color(color),
            )
        };
    }

    pub fn pipeline_render_create_info(&self) -> Option<ash::vk::PipelineRenderingCreateInfo> {
        match self.device_state.render_state {
            RenderState::Dynamic {
                ref color_attachments,
                depth_attachments,
                stencil_attachments,
            } => Some(
                ash::vk::PipelineRenderingCreateInfo::default()
                    .color_attachment_formats(color_attachments)
                    .depth_attachment_format(depth_attachments[0])
                    .stencil_attachment_format(stencil_attachments[0]),
            ),
            RenderState::Renderpass(_) => None,
        }
    }

    pub fn queue_ownership_transfer(&mut self, img: &BindlessImageResourceHandleEntryPair) {
        self.staging_system.queued_ownership_transfer.push((
            img.1.image,
            img.1.info.level_count,
            img.1.info.layer_count,
        ));
    }

    fn do_image_ownership_transfers(&mut self, cmd_buf: CommandBuffer) {
        if self.staging_system.queued_ownership_transfer.is_empty() {
            return;
        }

        let (qid_graphics, _, _, _, _) = self.queue_data(QueueType::Graphics);
        let (qid_transfer, _, _, _, _) = self.queue_data(QueueType::Transfer);

        let memory_barriers: SmallVec<[ImageMemoryBarrier2; 4]> = self
            .staging_system
            .queued_ownership_transfer
            .drain(..)
            .map(|(image, levels, layers)| {
                ImageMemoryBarrier2::default()
                    .image(image)
                    .dst_stage_mask(PipelineStageFlags2::ALL_GRAPHICS)
                    .dst_access_mask(AccessFlags2::SHADER_READ)
                    .src_queue_family_index(qid_transfer)
                    .dst_queue_family_index(qid_graphics)
                    .old_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .subresource_range(
                        ImageSubresourceRange::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(levels)
                            .base_array_layer(0)
                            .layer_count(layers),
                    )
            })
            .collect();

        unsafe {
            self.logical().cmd_pipeline_barrier2(
                cmd_buf,
                &DependencyInfo::default()
                    .dependency_flags(DependencyFlags::BY_REGION)
                    .image_memory_barriers(&memory_barriers),
            );
        }
    }

    pub fn wait_all_idle(&mut self) {
        let (_, queue, _, _, _) = self.queue_data(QueueType::Graphics);

        unsafe {
            self.device_state
                .logical
                .queue_wait_idle(queue)
                .expect("Failed to wait for idle queue");
            self.device_state
                .logical
                .device_wait_idle()
                .expect("Failed to wait for device idle");
        }
    }

    pub fn begin_rendering(&mut self, fb_size: ash::vk::Extent2D) -> FrameRenderContext {
        //
        // wait for previous submittted work
        unsafe {
            self.device_state
                .logical
                .wait_for_fences(
                    &[self.presentation_state.swapchain.work_fences
                        [self.presentation_state.frame_index as usize]],
                    true,
                    u64::MAX,
                )
                .expect("Failed to wait for submits");

            self.device_state
                .logical
                .reset_fences(&[self.presentation_state.swapchain.work_fences
                    [self.presentation_state.frame_index as usize]])
                .expect("Failed to reset fence ...");
        }

        let mut acquire_tries = 0u32;
        let acquired_image = 'acquire_swapchain_image: loop {
            let acquired_image = unsafe {
                self.presentation_state.swapchain_devext.acquire_next_image(
                    self.presentation_state.swapchain.swapchain,
                    u64::MAX,
                    self.presentation_state.swapchain.sem_image_available
                        [self.presentation_state.frame_index as usize],
                    Fence::null(),
                )
            };

            match acquired_image {
                Ok((image_idx, suboptimal)) => {
                    if !suboptimal {
                        break 'acquire_swapchain_image image_idx;
                    } else {
                        self.handle_suboptimal(fb_size);
                        acquire_tries += 1;
                    }
                }
                Err(acquire_err) => {
                    if acquire_tries >= 2 {
                        panic!("Failed to acquired new image after {acquire_tries} retries, crashing ...");
                    }
                    if acquire_err == ash::vk::Result::SUBOPTIMAL_KHR
                        || acquire_err == ash::vk::Result::ERROR_OUT_OF_DATE_KHR
                    {
                        self.handle_suboptimal(fb_size);
                        acquire_tries += 1;
                    } else {
                        panic!("Acquire image fatal error {acquire_err:?}");
                    }
                }
            };
        };

        //
        // begin command buffer + renderpass
        unsafe {
            self.device_state
                .logical
                .reset_command_buffer(
                    self.presentation_state.cmd_buffers
                        [self.presentation_state.frame_index as usize],
                    ash::vk::CommandBufferResetFlags::empty(),
                )
                .expect("Failed to reset command buffer");

            self.device_state
                .logical
                .begin_command_buffer(
                    self.presentation_state.cmd_buffers
                        [self.presentation_state.frame_index as usize],
                    &ash::vk::CommandBufferBeginInfo::default()
                        .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("Failed to begin command buffer ...");
        }

        //
        // ensure all loaded images are transfered to the graphics queue
        self.do_image_ownership_transfers(
            self.presentation_state.cmd_buffers[self.presentation_state.frame_index as usize],
        );

        let color_clear = ash::vk::ClearValue {
            color: ash::vk::ClearColorValue {
                float32: [0f32, 1f32, 0f32, 1f32],
            },
        };

        let depth_stencil_clear = ash::vk::ClearValue {
            depth_stencil: ash::vk::ClearDepthStencilValue {
                depth: 1.0f32,
                stencil: 0,
            },
        };

        let (qid, _queue, _cmd_pool, _, _) = self.queue_data(QueueType::Graphics);
        match self.device_state.render_state {
            RenderState::Dynamic { .. } => unsafe {
                //
                // transition attachments from undefined layout to optimal layout
                self.device_state.logical.cmd_pipeline_barrier2(
                    self.presentation_state.cmd_buffers
                        [self.presentation_state.frame_index as usize],
                    &DependencyInfo::default()
                        .dependency_flags(DependencyFlags::BY_REGION)
                        .image_memory_barriers(&[
                            ImageMemoryBarrier2::default()
                                .src_stage_mask(PipelineStageFlags2::TOP_OF_PIPE)
                                .src_access_mask(AccessFlags2::NONE)
                                .dst_stage_mask(PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                                .dst_access_mask(AccessFlags2::COLOR_ATTACHMENT_WRITE)
                                .old_layout(ImageLayout::UNDEFINED)
                                .new_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                                .src_queue_family_index(qid)
                                .dst_queue_family_index(qid)
                                .image(
                                    self.presentation_state.swapchain.images
                                        [acquired_image as usize],
                                )
                                .subresource_range(
                                    ImageSubresourceRange::default()
                                        .aspect_mask(ImageAspectFlags::COLOR)
                                        .base_mip_level(0)
                                        .level_count(1)
                                        .base_array_layer(0)
                                        .layer_count(1),
                                ),
                            ImageMemoryBarrier2::default()
                                .src_stage_mask(PipelineStageFlags2::TOP_OF_PIPE)
                                .src_access_mask(AccessFlags2::NONE)
                                .dst_stage_mask(PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                                .dst_access_mask(AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                                .old_layout(ImageLayout::UNDEFINED)
                                .new_layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                                .src_queue_family_index(qid)
                                .dst_queue_family_index(qid)
                                .image(
                                    self.presentation_state.swapchain.depth_stencil
                                        [acquired_image as usize]
                                        .0,
                                )
                                .subresource_range(
                                    ImageSubresourceRange::default()
                                        .aspect_mask(
                                            ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL,
                                        )
                                        .base_mip_level(0)
                                        .level_count(1)
                                        .base_array_layer(0)
                                        .layer_count(1),
                                ),
                        ]),
                );

                //
                // begin rendering
                self.device_state.logical.cmd_begin_rendering(
                    self.presentation_state.cmd_buffers
                        [self.presentation_state.frame_index as usize],
                    &ash::vk::RenderingInfo::default()
                        .render_area(Rect2D {
                            offset: Offset2D::default(),
                            extent: self.surface_state.caps.current_extent,
                        })
                        .layer_count(1)
                        .color_attachments(&[ash::vk::RenderingAttachmentInfo::default()
                            .image_view(
                                self.presentation_state.swapchain.image_views
                                    [acquired_image as usize],
                            )
                            .image_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(AttachmentLoadOp::CLEAR)
                            .store_op(AttachmentStoreOp::STORE)
                            .clear_value(color_clear)])
                        .depth_attachment(
                            &ash::vk::RenderingAttachmentInfo::default()
                                .image_view(
                                    self.presentation_state.swapchain.depth_stencil_views
                                        [acquired_image as usize],
                                )
                                .image_layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                                .load_op(AttachmentLoadOp::CLEAR)
                                .store_op(AttachmentStoreOp::DONT_CARE)
                                .clear_value(depth_stencil_clear),
                        )
                        .stencil_attachment(
                            &ash::vk::RenderingAttachmentInfo::default()
                                .image_view(
                                    self.presentation_state.swapchain.depth_stencil_views
                                        [acquired_image as usize],
                                )
                                .image_layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                                .load_op(AttachmentLoadOp::CLEAR)
                                .store_op(AttachmentStoreOp::DONT_CARE)
                                .clear_value(depth_stencil_clear),
                        ),
                );
            },
            RenderState::Renderpass(pass) => unsafe {
                self.device_state.logical.cmd_begin_render_pass(
                    self.presentation_state.cmd_buffers
                        [self.presentation_state.frame_index as usize],
                    &ash::vk::RenderPassBeginInfo::default()
                        .render_pass(pass)
                        .framebuffer(
                            self.presentation_state.swapchain.framebuffers[acquired_image as usize],
                        )
                        .render_area(Rect2D {
                            offset: Offset2D::default(),
                            extent: self.surface_state.caps.current_extent,
                        })
                        .clear_values(&[color_clear, depth_stencil_clear]),
                    ash::vk::SubpassContents::INLINE,
                );
            },
        }

        FrameRenderContext {
            cmd_buff: self.presentation_state.cmd_buffers
                [self.presentation_state.frame_index as usize],
            fb_size,
            current_frame_id: self.presentation_state.frame_index,
            acquired_swapchain_image: acquired_image,
        }
    }

    pub fn end_rendering(&mut self, frame_ctx: FrameRenderContext) {
        let (qid, queue, _cmd_pool, _, _) = self.queue_data(QueueType::Graphics);
        //
        // end command buffer + renderpass
        match self.render_state() {
            RenderState::Renderpass(_) => unsafe {
                self.device_state
                    .logical
                    .cmd_end_render_pass(frame_ctx.cmd_buff);
            },
            RenderState::Dynamic { .. } => unsafe {
                self.device_state
                    .logical
                    .cmd_end_rendering(frame_ctx.cmd_buff);
                //
                // transition image from attachment optimal to SRC_PRESENT
                self.device_state.logical.cmd_pipeline_barrier2(
                    frame_ctx.cmd_buff,
                    &DependencyInfo::default()
                        .dependency_flags(DependencyFlags::BY_REGION)
                        .image_memory_barriers(&[ImageMemoryBarrier2::default()
                            .src_stage_mask(PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                            .src_access_mask(AccessFlags2::COLOR_ATTACHMENT_WRITE)
                            .dst_stage_mask(PipelineStageFlags2::BOTTOM_OF_PIPE)
                            .dst_access_mask(AccessFlags2::MEMORY_READ)
                            .old_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .new_layout(ImageLayout::PRESENT_SRC_KHR)
                            .src_queue_family_index(qid)
                            .dst_queue_family_index(qid)
                            .image(
                                self.presentation_state.swapchain.images
                                    [frame_ctx.acquired_swapchain_image as usize],
                            )
                            .subresource_range(
                                ImageSubresourceRange::default()
                                    .aspect_mask(ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )]),
                );
            },
        }

        unsafe {
            self.device_state
                .logical
                .end_command_buffer(
                    self.presentation_state.cmd_buffers
                        [self.presentation_state.frame_index as usize],
                )
                .expect("Failed to end command buffer");
        }

        //
        // submit
        unsafe {
            self.device_state
                .logical
                .queue_submit2(
                    queue,
                    &[ash::vk::SubmitInfo2::default()
                        .wait_semaphore_infos(&[SemaphoreSubmitInfo::default()
                            .semaphore(
                                self.presentation_state.swapchain.sem_image_available
                                    [self.presentation_state.frame_index as usize],
                            )
                            .stage_mask(PipelineStageFlags2::TOP_OF_PIPE)])
                        .signal_semaphore_infos(&[SemaphoreSubmitInfo::default()
                            .semaphore(
                                self.presentation_state.swapchain.sem_work_done
                                    [self.presentation_state.frame_index as usize],
                            )
                            .stage_mask(PipelineStageFlags2::BOTTOM_OF_PIPE)])
                        .command_buffer_infos(&[CommandBufferSubmitInfo::default()
                            .command_buffer(
                                self.presentation_state.cmd_buffers
                                    [self.presentation_state.frame_index as usize],
                            )])],
                    self.presentation_state.swapchain.work_fences
                        [self.presentation_state.frame_index as usize],
                )
                .expect("Failed to submit work");

            let present_result = self.presentation_state.swapchain_devext.queue_present(
                queue,
                &ash::vk::PresentInfoKHR::default()
                    .image_indices(&[frame_ctx.acquired_swapchain_image])
                    .swapchains(&[self.presentation_state.swapchain.swapchain])
                    .wait_semaphores(&[self.presentation_state.swapchain.sem_work_done
                        [self.presentation_state.frame_index as usize]]),
            );

            self.presentation_state.frame_index =
                (self.presentation_state.frame_index + 1) % self.presentation_state.image_count;

            match present_result {
                Err(e) => {
                    if e == ash::vk::Result::ERROR_OUT_OF_DATE_KHR
                        || e == ash::vk::Result::SUBOPTIMAL_KHR
                    {
                        log::info!("Swapchain out of date, recreating ...");
                        self.handle_suboptimal(frame_ctx.fb_size);
                    } else {
                        log::error!("Present error: {:?}", e);
                        todo!("Handle this ...");
                    }
                }
                Ok(suboptimal) => {
                    if suboptimal {
                        log::info!("Swapchain suboptimal, recreating ...");
                        self.handle_suboptimal(frame_ctx.fb_size);
                    } else {
                        //
                        // nothing to do ...
                    }
                }
            }
        }
    }

    fn handle_suboptimal(&mut self, framebuffer_size: Extent2D) {
        let (_, graphics_queue, _, _, _) = self.queue_data(QueueType::Graphics);
        unsafe {
            self.logical()
                .queue_wait_idle(graphics_queue)
                .expect("Failed to wait idle on graphics queue");
        }

        let surface_caps = unsafe {
            self.surface_state
                .surface
                .ext
                .get_physical_device_surface_capabilities(
                    self.device_state.physical.device,
                    self.surface_state.surface.surface,
                )
        }
        .expect("Failed to query surface caps...");

        log::info!("Handle suboptimal, caps: {surface_caps:?}");

        let surface_extent = if surface_caps.current_extent.width == std::u32::MAX
            || surface_caps.current_extent.height == std::u32::MAX
        {
            framebuffer_size
        } else {
            surface_caps.current_extent
        };

        let mut swapchain = VulkanSwapchainState::create(&VulkanSwapchainCreateInfo {
            retired_swapchain: Some(self.presentation_state.swapchain.swapchain),
            surface: self.surface_state.surface.surface,
            image_count: self.presentation_state.image_count,
            image_format: self.surface_state.format,
            depth_format: self.surface_state.depth_format,
            extent: surface_extent,
            transform: surface_caps.current_transform,
            present_mode: self.surface_state.present_mode,
            ext: &self.presentation_state.swapchain_devext,
            device: self.logical(),
            memory_properties: self.memory_properties(),
            renderstate: &self.render_state(),
        })
        .expect("Failed to recreate swapchain");

        self.surface_state.caps.current_extent = surface_extent;
        std::mem::swap(&mut self.presentation_state.swapchain, &mut swapchain);
        swapchain.destroy(self.logical(), &self.presentation_state.swapchain_devext);
        self.presentation_state.frame_index = 0;
    }
}

struct PickedDeviceExtraData {
    surface_format: ash::vk::SurfaceFormatKHR,
    depth_stencil_format: ash::vk::Format,
    presentation_mode: ash::vk::PresentModeKHR,
    surface_caps: ash::vk::SurfaceCapabilitiesKHR,
    queue_family_graphics: u32,
    queue_family_transfer: u32,
}

fn pick_device(
    instance: &ash::Instance,
    surface_instance: &ash::khr::surface::Instance,
    surface_handle: ash::vk::SurfaceKHR,
) -> std::result::Result<(PhysicalDeviceState, PickedDeviceExtraData), GraphicsError> {
    unsafe { instance.enumerate_physical_devices() }?
        .into_iter()
        .filter_map(|phys_device| -> Option< (PhysicalDeviceState, PickedDeviceExtraData)  > {
            let (physical_device_properties, props_vk11, props_vk12, props_vk13) = unsafe {
                let mut props_vk11 = ash::vk::PhysicalDeviceVulkan11Properties::default();
                let mut props_vk12 = ash::vk::PhysicalDeviceVulkan12Properties::default();
                let mut props_vk13 = ash::vk::PhysicalDeviceVulkan13Properties::default();

                let mut phys_dev_props2 = ash::vk::PhysicalDeviceProperties2::default()
                    .push_next(&mut props_vk11)
                    .push_next(&mut props_vk12)
                    .push_next(&mut props_vk13);

                instance.get_physical_device_properties2(phys_device, &mut phys_dev_props2);
                (
                    phys_dev_props2.properties,
                    props_vk11,
                    props_vk12,
                    props_vk13,
                )
            };

            if physical_device_properties.device_type != ash::vk::PhysicalDeviceType::DISCRETE_GPU
            && physical_device_properties.device_type != ash::vk::PhysicalDeviceType::INTEGRATED_GPU
            {
                log::info!(
                    "Rejecting device {} (not a GPU device)",
                    physical_device_properties.device_id
                );
                return None;
            }

            let phys_dev_memory_props = unsafe {
                let mut props = ash::vk::PhysicalDeviceMemoryProperties2::default();
                instance.get_physical_device_memory_properties2(phys_device, &mut props);
                props.memory_properties
            };

            log::info!("{:?}", phys_dev_memory_props);

            log::info!(
                "{:?}\n{:?}\n{:?}\n{:?}",
                physical_device_properties,
                props_vk11,
                props_vk12,
                props_vk13
            );

            let (physical_device_features, f_vk11, f_vk12, f_vk13) = unsafe {
                let mut f_vk11 = ash::vk::PhysicalDeviceVulkan11Features::default();
                let mut f_vk12 = ash::vk::PhysicalDeviceVulkan12Features::default();
                let mut f_vk13 = ash::vk::PhysicalDeviceVulkan13Features::default();

                let mut pdf2 = ash::vk::PhysicalDeviceFeatures2::default()
                    .push_next(&mut f_vk11)
                    .push_next(&mut f_vk12)
                    .push_next(&mut f_vk13);

                instance.get_physical_device_features2(phys_device, &mut pdf2);

                (pdf2.features, f_vk11, f_vk12, f_vk13)
            };

            log::info!(
                "{:?}\n{:?}\n{:?}\n{:?}",
                physical_device_features,
                f_vk11,
                f_vk12,
                f_vk13
            );

            if physical_device_features.multi_draw_indirect == 0
                || physical_device_features.geometry_shader == 0
            {
                log::info!(
                    "Rejecting device {} (no geometry shader and/or MultiDrawIndirect)",
                    physical_device_properties.device_id
                );
                return None;
            }

            if f_vk12.descriptor_binding_partially_bound == 0
                || f_vk12.descriptor_indexing == 0
                || f_vk12.draw_indirect_count == 0
            {
                log::info!(
                    "Rejecting device {} (no descriptor partially bound/descriptor indexing/draw indirect count)",
                    physical_device_properties.device_id
                );
                return None;
             }

            let queue_family_props =
                unsafe { instance.get_physical_device_queue_family_properties(phys_device) };

            let maybe_queue_id = queue_family_props
                .iter()
                .enumerate()
                .find(|(_, queue_props)| {
                    queue_props
                        .queue_flags
                        .contains(ash::vk::QueueFlags::GRAPHICS | ash::vk::QueueFlags::COMPUTE)
                })
                .map(|(queue_id, _)| queue_id);

            if maybe_queue_id.is_none() {
                log::info!(
                    "Rejecting device {} (no GRAPHICS + COMPUTE support)",
                    physical_device_properties.device_id
                );
                return None;
            }

            let queue_id = maybe_queue_id.unwrap() as u32;
            let maybe_transfer_queue = queue_family_props
                .iter()
                .enumerate()
                .filter(|(qid, queue_props)| {
                    *qid != queue_id as usize
                        && queue_props
                            .queue_flags
                            .contains(ash::vk::QueueFlags::TRANSFER)
                })
                .map(|(qid, _)| qid)
                .nth(0);

            if maybe_transfer_queue.is_none() {
                log::info!(
                    "Rejecting device {} (no TRANSFER queue present)",
                    physical_device_properties.device_id,
                );
                return None;
            }
            //
            // query surface support
            match unsafe {
                surface_instance.get_physical_device_surface_support(phys_device, queue_id, surface_handle)
            } {
                Ok(true) => { Some(()) },
                Ok(false) => {
                    log::info!("Rejecting device {}, no surface support", physical_device_properties.device_id);
                    None
                },
                Err(e) => {
                    log::error!("Error querying device surface support {e:?}");
                    None
                }
            }?;

            let surface_caps = unsafe {
                surface_instance.get_physical_device_surface_capabilities(phys_device, surface_handle)
            }.ok()?;

            if !surface_caps
                .supported_usage_flags
                .intersects(ImageUsageFlags::COLOR_ATTACHMENT)
            {
                log::info!(
                    "Rejecting device {} (surface does not support COLOR_ATTACHMENT)",
                    physical_device_properties.device_id
                );
                return None;
             }

            // query surface formats
            let maybe_surface_format = unsafe {
                surface_instance
                    .get_physical_device_surface_formats(phys_device, surface_handle)
                    .map(|surface_formats| {
                        surface_formats
                            .iter()
                            .find(|fmt| {
                                fmt.format == Format::B8G8R8A8_UNORM
                                    || fmt.format == Format::R8G8B8A8_UNORM
                            })
                            .copied()
                    })
            }.ok().flatten();

            if maybe_surface_format.is_none() {
                log::info!("Rejecting device {} (does not support surface format B8G8R8A8_UNORM/R8G8B8A8_UNORM)", physical_device_properties.device_id);
                return None;
            }

            let image_format_props = unsafe {
                instance.get_physical_device_image_format_properties(
                    phys_device,
                    maybe_surface_format.unwrap().format,
                    ImageType::TYPE_2D,
                    ImageTiling::OPTIMAL,
                    ImageUsageFlags::SAMPLED,
                    ash::vk::ImageCreateFlags::empty(),
                )
            };

            if image_format_props.is_err() {
                log::info!(
                    "Rejecting device {} (image format properties error {:?})",
                    physical_device_properties.device_id,
                    image_format_props
                );
                return None;
            }

            //
            // query present mode
            let maybe_present_mode = unsafe {
                surface_instance .get_physical_device_surface_present_modes(phys_device, surface_handle)
                    .map(|present_modes| {
                        present_modes
                            .iter()
                            .find(|&&mode| {
                                mode == ash::vk::PresentModeKHR::MAILBOX || mode == ash::vk::PresentModeKHR::FIFO
                            })
                            .copied()
                    })
            }.ok().flatten();

            if maybe_present_mode.is_none() {
                log::info!(
                    "Rejecting device {} (does not support presentation mode MAILBOX/FIFO)",
                    physical_device_properties.device_id
                );
                return None;
            }

            //
            // query depth stencil support
            let depth_stencil_fmt = [Format::D32_SFLOAT_S8_UINT, Format::D24_UNORM_S8_UINT]
                .into_iter()
                .filter_map(|fmt| unsafe {
                    instance
                        .get_physical_device_image_format_properties(
                            phys_device,
                            fmt,
                            ImageType::TYPE_2D,
                            ImageTiling::OPTIMAL,
                            ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                            ash::vk::ImageCreateFlags::empty(),
                        )
                        .map(|_| fmt)
                        .ok()
                })
                .nth(0);

            if depth_stencil_fmt.is_none() {
                log::info!(
                    "Rejecting device {} (does not support depth stencil formats)",
                    physical_device_properties.device_id
                );
                return None;
            }

            let surface_caps= unsafe { surface_instance.get_physical_device_surface_capabilities(phys_device, surface_handle) }.ok()?;

            Some((
                    PhysicalDeviceState {
                        device: phys_device,
                        properties: PhysicalDeviceProperties { base: physical_device_properties },
                        features: PhysicalDeviceFeatures { base:  physical_device_features, dynamic_rendering: f_vk13.dynamic_rendering != 0, },
                        memory: phys_dev_memory_props,
                    },
                    PickedDeviceExtraData {
                        depth_stencil_format : depth_stencil_fmt.unwrap(),
                        surface_format: maybe_surface_format.unwrap(),
                        presentation_mode: maybe_present_mode.unwrap(),
                        queue_family_graphics: maybe_queue_id.unwrap() as u32,
                        queue_family_transfer: maybe_transfer_queue.unwrap() as u32,
                        surface_caps,
                    },
                    ))
        })
        .nth(0).ok_or_else(|| GraphicsError::Generic("No device with the required features was found".to_string()))
}

pub fn choose_memory_heap(
    memory_req: &ash::vk::MemoryRequirements,
    required_flags: ash::vk::MemoryPropertyFlags,
    memory_properties: &ash::vk::PhysicalDeviceMemoryProperties,
) -> u32 {
    for memory_type in 0..32 {
        if (memory_req.memory_type_bits & (1u32 << memory_type)) != 0 {
            if memory_properties.memory_types[memory_type]
                .property_flags
                .contains(required_flags)
            {
                return memory_type as u32;
            }
        }
    }

    panic!("Device does not support memory {:?}", required_flags);
}

#[derive(Copy, Clone, Debug)]
#[cfg(target_os = "linux")]
pub struct WindowSystemIntegration {
    pub native_disp: *mut std::os::raw::c_void,
    pub native_win: std::os::raw::c_ulong,
}

#[derive(Copy, Clone, Debug)]
#[cfg(target_os = "windows")]
pub struct WindowSystemIntegration {
    pub hwnd: isize,
    pub hinstance: isize,
}

#[cfg(target_os = "linux")]
fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    wsi: WindowSystemIntegration,
) -> std::result::Result<VulkanSurfaceWithInstance, GraphicsError> {
    let xlib_surface = ash::khr::xlib_surface::Instance::new(entry, instance);

    unsafe {
        xlib_surface.create_xlib_surface(
            &ash::vk::XlibSurfaceCreateInfoKHR::default()
                .dpy(wsi.native_disp as *mut ash::vk::Display)
                .window(std::mem::transmute::<u64, ash::vk::Window>(wsi.native_win)),
            None,
        )
    }
    .and_then(|khr_surface| {
        Ok(VulkanSurfaceWithInstance {
            ext: ash::khr::surface::Instance::new(entry, instance),
            surface: khr_surface,
        })
    })
    .map_err(|e| GraphicsError::from(e))
}

#[cfg(target_os = "windows")]
fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    wsi: WindowSystemIntegration,
) -> std::result::Result<VulkanSurfaceWithInstance, GraphicsError> {
    let win32_surface = ash::extensions::khr::Win32Surface::new(entry, instance);

    let khr_surface = unsafe {
        win32_surface.create_win32_surface(
            &ash::vk::Win32SurfaceCreateInfoKHR::builder()
                .hwnd(std::mem::transmute(wsi.hwnd))
                .hinstance(std::mem::transmute(wsi.hinstance)),
            None,
        )
    }?;

    Ok(VulkanSurfaceWithInstance {
        ext: ash::extensions::khr::Surface::new(entry, instance),
        surface: khr_surface,
    })
}

#[derive(Copy, Clone, Debug)]
pub enum RenderState {
    Renderpass(ash::vk::RenderPass),
    Dynamic {
        color_attachments: [Format; 1],
        depth_attachments: [Format; 1],
        stencil_attachments: [Format; 1],
    },
}

struct StagingSystem {
    staging_buffer: ash::vk::Buffer,
    staging_memory: ash::vk::DeviceMemory,
    mapped_memory: *mut std::ffi::c_void,
    free_ptr: std::sync::atomic::AtomicUsize,
    queued_ownership_transfer: Vec<(Image, u32, u32)>,
}

impl StagingSystem {
    fn create(
        device: &ash::Device,
        memory_properties: &ash::vk::PhysicalDeviceMemoryProperties,
    ) -> Result<StagingSystem, GraphicsError> {
        const STAGING_BUFFER_SIZE: u32 = 512 * 1024 * 2014;

        let staging_buffer = unsafe {
            device.create_buffer(
                &ash::vk::BufferCreateInfo::default()
                    .usage(ash::vk::BufferUsageFlags::TRANSFER_SRC)
                    .size(STAGING_BUFFER_SIZE as u64)
                    .sharing_mode(SharingMode::EXCLUSIVE),
                None,
            )
        }?;

        scopeguard::defer_on_unwind! {
            unsafe {
                device.destroy_buffer(staging_buffer, None);
            }
        }

        let mem_req = unsafe { device.get_buffer_memory_requirements(staging_buffer) };
        let staging_memory = unsafe {
            device.allocate_memory(
                &MemoryAllocateInfo::default()
                    .allocation_size(mem_req.size)
                    .memory_type_index(choose_memory_heap(
                        &mem_req,
                        MemoryPropertyFlags::HOST_COHERENT,
                        memory_properties,
                    )),
                None,
            )
        }?;

        scopeguard::defer_on_unwind! {
            unsafe {
                device.free_memory(staging_memory, None);
            }
        }

        unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0) }?;

        let mapped_memory = unsafe {
            device.map_memory(
                staging_memory,
                0,
                STAGING_BUFFER_SIZE as ash::vk::DeviceSize,
                ash::vk::MemoryMapFlags::empty(),
            )
        }?;

        Ok(StagingSystem {
            staging_buffer,
            staging_memory,
            mapped_memory,
            free_ptr: std::sync::atomic::AtomicUsize::new(0),
            queued_ownership_transfer: vec![],
        })
    }

    fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.unmap_memory(self.staging_memory);
            device.free_memory(self.staging_memory, None);
            device.destroy_buffer(self.staging_buffer, None);
        }
    }
}

pub struct QueueSubmitWaitToken {
    pub cmd_buffer: ash::vk::CommandBuffer,
    wait_fence: ash::vk::Fence,
    queue_type: QueueType,
}

pub struct QueuedJob {
    pub cmd_buffer: ash::vk::CommandBuffer,
    pub queue_type: QueueType,
}

pub struct VulkanRenderer_DebugQueueScope<'a> {
    logical: &'a VulkanRenderer,
}

impl<'a> VulkanRenderer_DebugQueueScope<'a> {
    pub fn begin(renderer: &'a VulkanRenderer, name: &str, color: [f32; 4]) -> Self {
        renderer.debug_queue_begin_label(name, color);
        Self { logical: renderer }
    }
}

impl<'a> std::ops::Drop for VulkanRenderer_DebugQueueScope<'a> {
    fn drop(&mut self) {
        self.logical.debug_queue_end_label();
    }
}

pub struct VulkanRenderer_DebugMarkerScope<'a> {
    renderer: &'a VulkanRenderer,
    cmd_buffer: CommandBuffer,
}

impl<'a> VulkanRenderer_DebugMarkerScope<'a> {
    pub fn begin(
        renderer: &'a VulkanRenderer,
        cmd_buffer: CommandBuffer,
        name: &str,
        color: [f32; 4],
    ) -> Self {
        renderer.debug_marker_begin(cmd_buffer, name, color);
        Self {
            renderer,
            cmd_buffer,
        }
    }
}

impl<'a> std::ops::Drop for VulkanRenderer_DebugMarkerScope<'a> {
    fn drop(&mut self) {
        self.renderer.debug_marker_end(self.cmd_buffer);
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(u8)]
pub enum QueueType {
    Graphics,
    Transfer,
}

struct PresentationState {
    cmd_pool: ash::vk::CommandPool,
    swapchain_devext: ash::khr::swapchain::Device,
    swapchain: VulkanSwapchainState,
    frame_index: u32,
    acquired_image: u32,
    image_count: u32,
    cmd_buffers: Vec<CommandBuffer>,
}

impl PresentationState {
    fn create(
        device: &ash::Device,
        swapchain_devext: ash::khr::swapchain::Device,
        swapchain: VulkanSwapchainState,
        image_count: u32,
    ) -> std::result::Result<PresentationState, GraphicsError> {
        let cmd_pool = unsafe {
            device
                .create_command_pool(
                    &ash::vk::CommandPoolCreateInfo::default()
                        .queue_family_index(0)
                        .flags(
                            ash::vk::CommandPoolCreateFlags::TRANSIENT
                                | ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                        ),
                    None,
                )
                .map_err(|e| GraphicsError::from(e))
        }?;

        let cmd_buffers: Vec<CommandBuffer> = unsafe {
            device.allocate_command_buffers(
                &ash::vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .command_buffer_count(image_count)
                    .level(ash::vk::CommandBufferLevel::PRIMARY),
            )
        }?;

        Ok(PresentationState {
            cmd_pool,
            swapchain_devext,
            swapchain,
            frame_index: 0u32,
            acquired_image: 0u32,
            image_count,
            cmd_buffers,
        })
    }

    fn destroy(&mut self, device: &ash::Device) {
        self.swapchain.destroy(device, &self.swapchain_devext);
        unsafe {
            device.destroy_command_pool(self.cmd_pool, None);
        }
    }
}

pub struct VulkanSwapchainState {
    pub swapchain: ash::vk::SwapchainKHR,
    pub images: Vec<Image>,
    pub image_views: Vec<ImageView>,
    pub framebuffers: Vec<Framebuffer>,
    pub depth_stencil: Vec<(Image, DeviceMemory)>,
    pub depth_stencil_views: Vec<ImageView>,
    pub work_fences: Vec<Fence>,
    pub sem_work_done: Vec<Semaphore>,
    pub sem_image_available: Vec<Semaphore>,
}

struct VulkanSwapchainCreateInfo<'a> {
    retired_swapchain: Option<ash::vk::SwapchainKHR>,
    surface: ash::vk::SurfaceKHR,
    image_count: u32,
    image_format: ash::vk::SurfaceFormatKHR,
    depth_format: ash::vk::Format,
    extent: ash::vk::Extent2D,
    transform: ash::vk::SurfaceTransformFlagsKHR,
    present_mode: ash::vk::PresentModeKHR,
    ext: &'a ash::khr::swapchain::Device,
    device: &'a ash::Device,
    memory_properties: &'a ash::vk::PhysicalDeviceMemoryProperties,
    renderstate: &'a RenderState,
}

impl VulkanSwapchainState {
    fn create(
        create_info: &VulkanSwapchainCreateInfo,
    ) -> std::result::Result<VulkanSwapchainState, GraphicsError> {
        log::info!(
            "Creating swapchain, retired swapchain {:?}",
            create_info.retired_swapchain
        );

        let swapchain = unsafe {
            create_info.ext.create_swapchain(
                &ash::vk::SwapchainCreateInfoKHR::default()
                    .surface(create_info.surface)
                    .min_image_count(create_info.image_count)
                    .image_format(create_info.image_format.format)
                    .image_color_space(create_info.image_format.color_space)
                    .image_extent(create_info.extent)
                    .image_array_layers(1)
                    .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
                    .image_sharing_mode(SharingMode::EXCLUSIVE)
                    .composite_alpha(ash::vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .pre_transform(create_info.transform)
                    .old_swapchain(
                        create_info
                            .retired_swapchain
                            .unwrap_or_else(|| ash::vk::SwapchainKHR::null()),
                    )
                    .present_mode(create_info.present_mode),
                None,
            )
        }?;

        log::info!("Created swapchain {swapchain:p}");

        let images = unsafe { create_info.ext.get_swapchain_images(swapchain) }?;
        let image_count = images.len();

        let image_views: Vec<ImageView> = Result::from_iter(
            images
                .iter()
                .map(|img| unsafe {
                    create_info.device.create_image_view(
                        &ImageViewCreateInfo::default()
                            .image(*img)
                            .view_type(ImageViewType::TYPE_2D)
                            .format(create_info.image_format.format)
                            .components(ash::vk::ComponentMapping::default())
                            .subresource_range(
                                ImageSubresourceRange::default()
                                    .aspect_mask(ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            ),
                        None,
                    )
                })
                .collect::<Vec<_>>(),
        )?;

        let framebuffers: Vec<ash::vk::Framebuffer> = match create_info.renderstate {
            RenderState::Dynamic { .. } => vec![],
            RenderState::Renderpass(pass) => Result::from_iter(
                image_views
                    .iter()
                    .map(|&img_view| unsafe {
                        create_info.device.create_framebuffer(
                            &ash::vk::FramebufferCreateInfo::default()
                                .render_pass(*pass)
                                .attachments(&[img_view])
                                .width(create_info.extent.width)
                                .height(create_info.extent.height)
                                .layers(1),
                            None,
                        )
                    })
                    .collect::<Vec<_>>(),
            )?,
        };

        //
        // create depth stencil images and image views
        let depth_stencil: Vec<(Image, DeviceMemory)> = Result::from_iter(
            image_views
                .iter()
                .map(|_| -> Result<(Image, DeviceMemory), ash::vk::Result> {
                    let depth_stencil_image = unsafe {
                        create_info.device.create_image(
                            &ImageCreateInfo::default()
                                .image_type(ImageType::TYPE_2D)
                                .format(create_info.depth_format)
                                .extent(Extent3D {
                                    width: create_info.extent.width,
                                    height: create_info.extent.height,
                                    depth: 1,
                                })
                                .mip_levels(1)
                                .array_layers(1)
                                .samples(ash::vk::SampleCountFlags::TYPE_1)
                                .tiling(ImageTiling::OPTIMAL)
                                .usage(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                                .sharing_mode(SharingMode::EXCLUSIVE)
                                .initial_layout(ImageLayout::UNDEFINED),
                            None,
                        )
                    }?;

                    let image_mem_req = unsafe {
                        create_info
                            .device
                            .get_image_memory_requirements(depth_stencil_image)
                    };

                    let depth_stencil_memory = unsafe {
                        create_info.device.allocate_memory(
                            &MemoryAllocateInfo::default()
                                .allocation_size(image_mem_req.size)
                                .memory_type_index(choose_memory_heap(
                                    &image_mem_req,
                                    MemoryPropertyFlags::DEVICE_LOCAL,
                                    create_info.memory_properties,
                                )),
                            None,
                        )
                    }?;

                    unsafe {
                        create_info.device.bind_image_memory(
                            depth_stencil_image,
                            depth_stencil_memory,
                            0,
                        )?;
                    }

                    Ok((depth_stencil_image, depth_stencil_memory))
                })
                .collect::<Vec<_>>(),
        )?;

        let depth_stencil_views: Vec<ImageView> = Result::from_iter(
            depth_stencil
                .iter()
                .map(|&(image, _)| unsafe {
                    create_info.device.create_image_view(
                        &ImageViewCreateInfo::default()
                            .image(image)
                            .view_type(ImageViewType::TYPE_2D)
                            .format(create_info.depth_format)
                            .components(ash::vk::ComponentMapping::default())
                            .subresource_range(
                                ImageSubresourceRange::default()
                                    .aspect_mask(
                                        ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL,
                                    )
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            ),
                        None,
                    )
                })
                .collect::<Vec<_>>(),
        )?;

        Ok(VulkanSwapchainState {
            swapchain,
            images,
            image_views,
            framebuffers,
            depth_stencil,
            depth_stencil_views,

            work_fences: Result::from_iter(
                (0..image_count)
                    .map(|_| unsafe {
                        create_info.device.create_fence(
                            &ash::vk::FenceCreateInfo::default()
                                .flags(ash::vk::FenceCreateFlags::SIGNALED),
                            None,
                        )
                    })
                    .collect::<Vec<_>>(),
            )?,

            sem_work_done: Result::from_iter(
                (0..image_count)
                    .map(|_| unsafe {
                        create_info
                            .device
                            .create_semaphore(&ash::vk::SemaphoreCreateInfo::default(), None)
                    })
                    .collect::<Vec<_>>(),
            )?,

            sem_image_available: Result::from_iter(
                (0..image_count)
                    .map(|_| unsafe {
                        create_info
                            .device
                            .create_semaphore(&ash::vk::SemaphoreCreateInfo::default(), None)
                    })
                    .collect::<Vec<_>>(),
            )?,
        })
    }

    fn destroy(&mut self, device: &ash::Device, swapchain_ext: &ash::khr::swapchain::Device) {
        self.image_views
            .iter()
            .for_each(|&view| unsafe { device.destroy_image_view(view, None) });
        self.framebuffers
            .iter()
            .for_each(|&fb| unsafe { device.destroy_framebuffer(fb, None) });
        self.depth_stencil.iter().for_each(|(ds, mem)| unsafe {
            device.destroy_image(*ds, None);
            device.free_memory(*mem, None);
        });
        self.depth_stencil_views.iter().for_each(|&view| unsafe {
            device.destroy_image_view(view, None);
        });
        self.work_fences
            .iter()
            .for_each(|&fence| unsafe { device.destroy_fence(fence, None) });
        self.sem_work_done
            .iter()
            .chain(self.sem_image_available.iter())
            .for_each(|&sem| unsafe {
                device.destroy_semaphore(sem, None);
            });
        unsafe {
            if !self.swapchain.is_null() {
                swapchain_ext.destroy_swapchain(self.swapchain, None);
            }
        }
    }
}

#[derive(Copy, Clone)]
pub struct FrameRenderContext {
    pub cmd_buff: ash::vk::CommandBuffer,
    pub fb_size: ash::vk::Extent2D,
    pub current_frame_id: u32,
    pub acquired_swapchain_image: u32,
}

//
// pub mod misc {
//     pub fn write_ppm<P: AsRef<std::path::Path>>(file: P, width: u32, height: u32, pixels: &[u8]) {
//         use std::io::Write;
//         let mut f = std::fs::File::create(file).unwrap();
//
//         writeln!(&mut f, "P3 {width} {height} 255").unwrap();
//         pixels.chunks(4).for_each(|c| {
//             writeln!(&mut f, "{} {} {}", c[0], c[1], c[2]).unwrap();
//         });
//     }
// }
//
