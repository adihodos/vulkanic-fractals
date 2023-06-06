use std::ffi::{c_char, c_void, CString};

use ash::{
    extensions::ext::DebugUtils,
    vk::{
        AccessFlags, ApplicationInfo, AttachmentDescription, AttachmentLoadOp, AttachmentReference,
        AttachmentStoreOp, Buffer, BufferCreateInfo, BufferImageCopy, BufferUsageFlags,
        CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferResetFlags,
        CommandBufferUsageFlags, CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo,
        ComponentMapping, ComponentSwizzle, CompositeAlphaFlagsKHR,
        DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
        DebugUtilsMessengerCreateInfoEXT, DebugUtilsMessengerEXT, DependencyFlags, DescriptorPool,
        DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSetLayout,
        DescriptorSetLayoutCreateInfo, DescriptorType, DeviceCreateInfo, DeviceMemory,
        DeviceQueueCreateInfo, DeviceSize, Extent2D, Fence, FenceCreateFlags, FenceCreateInfo,
        Format, Framebuffer, FramebufferCreateInfo, GraphicsPipelineCreateInfo, Image,
        ImageAspectFlags, ImageCreateInfo, ImageLayout, ImageMemoryBarrier, ImageSubresourceLayers,
        ImageSubresourceRange, ImageUsageFlags, ImageView, ImageViewCreateInfo, ImageViewType,
        InstanceCreateInfo, MappedMemoryRange, MemoryAllocateInfo, MemoryMapFlags,
        MemoryPropertyFlags, MemoryRequirements, PhysicalDevice, PhysicalDeviceFeatures,
        PhysicalDeviceMemoryProperties, PhysicalDeviceProperties, PhysicalDeviceType, Pipeline,
        PipelineBindPoint, PipelineCache, PipelineLayout, PipelineLayoutCreateInfo,
        PipelineStageFlags, PresentInfoKHR, PresentModeKHR, Queue, RenderPass,
        RenderPassCreateInfo, SampleCountFlags, Sampler, SamplerCreateInfo, Semaphore,
        SemaphoreCreateInfo, ShaderModule, ShaderModuleCreateInfo, SharingMode, SubmitInfo,
        SubpassDescription, SurfaceFormatKHR, SurfaceKHR, SurfaceTransformFlagsKHR,
        SwapchainCreateInfoKHR, SwapchainKHR,
    },
    Device, Entry, Instance,
};

pub struct UniqueImage {
    device: *const Device,
    pub image: Image,
    pub memory: DeviceMemory,
}

impl UniqueImage {
    pub fn from_bytes(
        vks: &mut VulkanState,
        create_info: ImageCreateInfo,
        pixels: &[u8],
    ) -> UniqueImage {
        let image = unsafe { vks.ds.device.create_image(&create_info, None) }
            .expect("Failed to create image");

        let memory_req = unsafe { vks.ds.device.get_image_memory_requirements(image) };

        let image_memory = unsafe {
            vks.ds.device.allocate_memory(
                &MemoryAllocateInfo::builder()
                    .allocation_size(memory_req.size)
                    .memory_type_index(choose_memory_heap(
                        &memory_req,
                        MemoryPropertyFlags::DEVICE_LOCAL,
                        &vks.ds,
                    )),
                None,
            )
        }
        .expect("Failed to allocate memory");

        unsafe { vks.ds.device.bind_image_memory(image, image_memory, 0) }.expect(&format!(
            "Failed to bind memory @ {:?} for image {:?}",
            image, image_memory
        ));

        let img = UniqueImage {
            device: &vks.ds.device as *const _,
            image,
            memory: image_memory,
        };

        vks.copy_pixels_to_image(&img, pixels, &create_info);

        img
    }
}

impl std::ops::Drop for UniqueImage {
    fn drop(&mut self) {
        unsafe {
            (*self.device).destroy_image(self.image, None);
            (*self.device).free_memory(self.memory, None);
        }
    }
}

pub struct UniqueImageView {
    device: *const Device,
    pub view: ImageView,
    pub image: Image,
}

impl UniqueImageView {
    pub fn new(
        vks: &VulkanState,
        img: &UniqueImage,
        img_view_create_info: ImageViewCreateInfo,
    ) -> UniqueImageView {
        let view = unsafe { vks.ds.device.create_image_view(&img_view_create_info, None) }
            .expect("Failed to create image view");

        UniqueImageView {
            device: &vks.ds.device as *const _,
            view,
            image: img.image,
        }
    }
}

pub struct UniqueSampler {
    device: *const Device,
    pub handle: Sampler,
}

impl UniqueSampler {
    pub fn new(vks: &VulkanState, create_info: SamplerCreateInfo) -> UniqueSampler {
        let handle = unsafe { vks.ds.device.create_sampler(&create_info, None) }
            .expect("Failed to create sampler");

        UniqueSampler {
            device: &vks.ds.device as *const _,
            handle,
        }
    }
}

pub struct UniquePipeline {
    device: *const Device,
    pub handle: Pipeline,
    pub layout: PipelineLayout,
    pub descriptor_set_layout: Vec<DescriptorSetLayout>,
}

impl std::ops::Drop for UniquePipeline {
    fn drop(&mut self) {
        self.descriptor_set_layout
            .iter()
            .for_each(|&ds_layout| unsafe {
                (*self.device).destroy_descriptor_set_layout(ds_layout, None);
            });

        unsafe {
            (*self.device).destroy_pipeline_layout(self.layout, None);
            (*self.device).destroy_pipeline(self.handle, None);
        }
    }
}

impl UniquePipeline {
    pub fn new(
        vks: &VulkanState,
        descriptor_set_layout_info: &[DescriptorSetLayoutCreateInfo],
        pipeline_create_info: GraphicsPipelineCreateInfo,
    ) -> UniquePipeline {
        let descriptor_set_layout = descriptor_set_layout_info
            .iter()
            .map(|ds_layout| unsafe {
                vks.ds
                    .device
                    .create_descriptor_set_layout(&ds_layout, None)
                    .expect("Failed to create descriptor set layout")
            })
            .collect::<Vec<_>>();

        let layout = unsafe {
            vks.ds.device.create_pipeline_layout(
                &PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layout),
                None,
            )
        }
        .expect("Failed to create pipeline layout");

        let mut pipeline_create_info = pipeline_create_info;
        pipeline_create_info.layout = layout;

        let p_handles = unsafe {
            vks.ds.device.create_graphics_pipelines(
                PipelineCache::null(),
                &[pipeline_create_info],
                None,
            )
        }
        .expect("Failed to create pipeline");

        UniquePipeline {
            device: &vks.ds.device as *const _,
            handle: p_handles[0],
            layout,
            descriptor_set_layout,
        }
    }
}

pub fn choose_memory_heap(
    memory_req: &MemoryRequirements,
    required_flags: MemoryPropertyFlags,
    ds: &VulkanDeviceState,
) -> u32 {
    for memory_type in 0..32 {
        if (memory_req.memory_type_bits & (1u32 << memory_type)) != 0 {
            if ds.physical.memory_properties.memory_types[memory_type]
                .property_flags
                .contains(required_flags)
            {
                return memory_type as u32;
            }
        }
    }

    log::error!("Device does not support memory {:?}", required_flags);
    panic!("Ay blyat");
}

pub struct UniqueShaderModule {
    device: *const ash::Device,
    module: ShaderModule,
}

impl std::ops::Deref for UniqueShaderModule {
    type Target = ShaderModule;

    fn deref(&self) -> &Self::Target {
        &self.module
    }
}

impl std::ops::Drop for UniqueShaderModule {
    fn drop(&mut self) {
        unsafe {
            (*self.device).destroy_shader_module(self.module, None);
        }
    }
}

pub fn compile_shader_from_file<P: AsRef<std::path::Path>>(
    p: P,
    vk_dev: &ash::Device,
) -> Option<UniqueShaderModule> {
    use shaderc::*;

    let mut compile_options = CompileOptions::new().unwrap();
    compile_options.set_source_language(SourceLanguage::GLSL);
    compile_options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_2 as u32);
    compile_options.set_generate_debug_info();
    compile_options.set_warnings_as_errors();

    let shader_type = p
        .as_ref()
        .extension()
        .map(|ext| ext.to_str())
        .flatten()
        .map(|ext| match ext {
            "vert" => ShaderKind::Vertex,
            "frag" => ShaderKind::Fragment,
            "geom" => ShaderKind::Geometry,
            _ => todo!("Shader type not handled"),
        })
        .unwrap();

    let src_code = std::fs::read_to_string(&p).expect(&format!(
        "Failed to read shader file {}",
        p.as_ref().display()
    ));

    let compiler = Compiler::new().unwrap();
    let compile_result = compiler.compile_into_spirv(
        &src_code,
        shader_type,
        p.as_ref().as_os_str().to_str().unwrap(),
        "main",
        Some(&compile_options),
    );

    compile_result
        .map(|compiled_code| {
            let module = unsafe {
                vk_dev.create_shader_module(
                    &ShaderModuleCreateInfo::builder().code(compiled_code.as_binary()),
                    None,
                )
            }
            .expect("Failed to create shader module from bytecode ...");
            UniqueShaderModule {
                module,
                device: vk_dev as *const _,
            }
        })
        .map_err(|e| {
            log::error!(
                "Failed to compile shader {}, error:\n{}",
                p.as_ref().as_os_str().to_str().unwrap(),
                e.to_string()
            );
        })
        .ok()
}

pub struct UniqueBuffer {
    device: *const Device,
    pub handle: Buffer,
    pub memory: DeviceMemory,
    pub item_aligned_size: usize,
}

impl std::ops::Drop for UniqueBuffer {
    fn drop(&mut self) {
        log::debug!(
            "Dropping buffer and memory {:?} -> {:?}",
            self.handle,
            self.memory
        );
        unsafe {
            (*self.device).destroy_buffer(self.handle, None);
            (*self.device).free_memory(self.memory, None);
        }
    }
}

impl UniqueBuffer {
    pub fn new<T: Sized>(
        ds: &VulkanState,
        usage: BufferUsageFlags,
        memory_flags: MemoryPropertyFlags,
        items: usize,
    ) -> Self {
        let align_size = if usage.intersects(BufferUsageFlags::UNIFORM_BUFFER) {
            ds.ds
                .physical
                .properties
                .limits
                .min_uniform_buffer_offset_alignment
        } else if usage.intersects(
            BufferUsageFlags::UNIFORM_TEXEL_BUFFER | BufferUsageFlags::STORAGE_TEXEL_BUFFER,
        ) {
            ds.ds
                .physical
                .properties
                .limits
                .min_texel_buffer_offset_alignment
        } else if usage.intersects(BufferUsageFlags::STORAGE_BUFFER) {
            ds.ds
                .physical
                .properties
                .limits
                .min_storage_buffer_offset_alignment
        } else {
            ds.ds.physical.properties.limits.non_coherent_atom_size
        } as usize;

        let item_aligned_size = round_up(std::mem::size_of::<T>(), align_size);
        let size = item_aligned_size * items;

        let buffer = unsafe {
            ds.ds.device.create_buffer(
                &BufferCreateInfo::builder()
                    .size(size as DeviceSize)
                    .usage(usage)
                    .sharing_mode(SharingMode::EXCLUSIVE)
                    .queue_family_indices(&[ds.ds.physical.queue_family_id]),
                None,
            )
        }
        .expect("Failed to create buffer");

        let memory_req = unsafe { ds.ds.device.get_buffer_memory_requirements(buffer) };
        let mem_heap = choose_memory_heap(&memory_req, memory_flags, &ds.ds);

        let buffer_memory = unsafe {
            ds.ds.device.allocate_memory(
                &MemoryAllocateInfo::builder()
                    .allocation_size(memory_req.size)
                    .memory_type_index(mem_heap),
                None,
            )
        }
        .expect("Failed to allocate memory for buffer ...");

        unsafe {
            ds.ds
                .device
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .expect("Failed to bind memory for buffer");
        }

        log::debug!(
            "Create buffer and memory object {:?} -> {:?}",
            buffer,
            buffer_memory
        );

        Self {
            device: &ds.ds.device as *const _,
            handle: buffer,
            memory: buffer_memory,
            item_aligned_size,
        }
    }
}

fn round_up(num_to_round: usize, multiple: usize) -> usize {
    assert_ne!(multiple, 0);
    ((num_to_round + multiple - 1) / multiple) * multiple
}

pub struct UniqueBufferMapping<'a> {
    buffer: Buffer,
    buffer_mem: DeviceMemory,
    mapped_memory: *mut c_void,
    offset: usize,
    size: usize,
    alignment: usize,
    device: &'a Device,
}

impl<'a> UniqueBufferMapping<'a> {
    pub fn new(
        buf: &UniqueBuffer,
        ds: &'a VulkanDeviceState,
        offset: Option<usize>,
        size: Option<usize>,
    ) -> Self {
        let offset = offset.unwrap_or_default();
        if offset != 0 && size.is_none() {
            log::error!(
                "When mapping a buffer and specifying an offset a size must also be provided"
            );
            panic!("blyat!!");
        }

        let size = size.unwrap_or(ash::vk::WHOLE_SIZE as usize);

        let mapped_memory = unsafe {
            ds.device.map_memory(
                buf.memory,
                offset as u64,
                size as u64,
                MemoryMapFlags::empty(),
            )
        }
        .expect("Failed to map memory");

        Self {
            buffer: buf.handle,
            buffer_mem: buf.memory,
            mapped_memory,
            device: &ds.device,
            offset,
            alignment: buf.item_aligned_size,
            size,
        }
    }

    pub fn write_data<T: Sized + Copy>(&mut self, data: &[T]) {
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.mapped_memory as *mut T, data.len());
        }
    }

    pub fn write_data_with_offset<T: Sized + Copy>(&mut self, data: &[T], offset: isize) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                (self.mapped_memory as *mut T).offset(offset),
                data.len(),
            );
        }
    }
}

impl<'a> std::ops::Drop for UniqueBufferMapping<'a> {
    fn drop(&mut self) {
        unsafe {
            self.device
                .flush_mapped_memory_ranges(&[*MappedMemoryRange::builder()
                    .memory(self.buffer_mem)
                    .offset(self.offset as DeviceSize)
                    .size(self.size as DeviceSize)])
                .expect("Failed to flush mapped memory ...");

            self.device.unmap_memory(self.buffer_mem);
        }
    }
}

struct ResourceLoadingState {
    cmd_buf: CommandBuffer,
    fence: Fence,
    work_buffers: Vec<UniqueBuffer>,
}

impl ResourceLoadingState {
    fn new(ds: &VulkanDeviceState) -> ResourceLoadingState {
        let cmd_buf = unsafe {
            ds.device.allocate_command_buffers(
                &CommandBufferAllocateInfo::builder()
                    .command_buffer_count(1)
                    .command_pool(ds.cmd_pool)
                    .level(ash::vk::CommandBufferLevel::PRIMARY),
            )
        }
        .expect("Failed to allocate command buffer")[0];

        let fence = unsafe { ds.device.create_fence(&FenceCreateInfo::builder(), None) }
            .expect("Failed to create fence");

        ResourceLoadingState {
            cmd_buf,
            fence,
            work_buffers: vec![],
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct WindowSystemIntegration {
    pub native_disp: *mut std::os::raw::c_void,
    pub native_win: std::os::raw::c_ulong,
}

pub struct VulkanState {
    resource_loader: ResourceLoadingState,
    pub renderpass: RenderPass,
    pub swapchain: VulkanSwapchainState,
    pub ds: VulkanDeviceState,
    pub msgr: DebugUtilsMessengerEXT,
    pub dbg: DebugUtils,
    pub instance: Instance,
    pub entry: Entry,
}

impl VulkanState {
    pub fn wait_all_idle(&mut self) {
        unsafe {
            self.ds
                .device
                .queue_wait_idle(self.ds.queue)
                .expect("Failed to wait for idle queue");
            self.ds
                .device
                .device_wait_idle()
                .expect("Failed to wait for device idle");
        }
    }

    pub fn begin_resource_loading(&self) {
        unsafe {
            self.ds.device.begin_command_buffer(
                self.resource_loader.cmd_buf,
                &CommandBufferBeginInfo::builder().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .expect("Failed to begin command buffer for resource loading");
    }

    pub fn end_resource_loading(&mut self) {
        unsafe {
            self.ds
                .device
                .end_command_buffer(self.resource_loader.cmd_buf)
                .expect("Failed to end command buffer");
            self.ds
                .device
                .queue_submit(
                    self.ds.queue,
                    &[*SubmitInfo::builder().command_buffers(&[self.resource_loader.cmd_buf])],
                    self.resource_loader.fence,
                )
                .expect("Failed to submit command buffer");
            self.ds
                .device
                .wait_for_fences(&[self.resource_loader.fence], true, u64::MAX)
                .expect("Failed to wait for fences ...");
        }

        self.resource_loader.work_buffers.clear();
    }

    pub fn copy_pixels_to_image(
        &mut self,
        img: &UniqueImage,
        pixels: &[u8],
        image_info: &ImageCreateInfo,
    ) {
        let work_buffer = UniqueBuffer::new::<u8>(
            self,
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_SRC,
            MemoryPropertyFlags::HOST_VISIBLE,
            pixels.len(),
        );

        UniqueBufferMapping::new(&work_buffer, &self.ds, None, None).write_data(pixels);

        let img_subresource_range = *ImageSubresourceRange::builder()
            .aspect_mask(ImageAspectFlags::COLOR)
            .layer_count(image_info.array_layers)
            .base_array_layer(0)
            .level_count(image_info.mip_levels)
            .base_mip_level(0);

        //
        // transition image layout from undefined -> transfer src
        unsafe {
            self.ds.device.cmd_pipeline_barrier(
                self.resource_loader.cmd_buf,
                PipelineStageFlags::TOP_OF_PIPE,
                PipelineStageFlags::TRANSFER,
                DependencyFlags::empty(),
                &[],
                &[],
                &[*ImageMemoryBarrier::builder()
                    .src_access_mask(AccessFlags::NONE)
                    .dst_access_mask(AccessFlags::TRANSFER_WRITE)
                    .image(img.image)
                    .old_layout(ImageLayout::UNDEFINED)
                    .new_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
                    .subresource_range(img_subresource_range)],
            );
        }

        //
        // copy pixels
        unsafe {
            self.ds.device.cmd_copy_buffer_to_image(
                self.resource_loader.cmd_buf,
                work_buffer.handle,
                img.image,
                ImageLayout::TRANSFER_DST_OPTIMAL,
                &[*BufferImageCopy::builder()
                    .buffer_offset(0)
                    .image_subresource(
                        *ImageSubresourceLayers::builder()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .base_array_layer(0)
                            .layer_count(image_info.array_layers)
                            .mip_level(0),
                    )
                    .image_extent(image_info.extent)],
            );
        }

        //
        // transition layout from transfer -> shader readonly optimal
        unsafe {
            self.ds.device.cmd_pipeline_barrier(
                self.resource_loader.cmd_buf,
                PipelineStageFlags::TRANSFER,
                PipelineStageFlags::FRAGMENT_SHADER,
                DependencyFlags::empty(),
                &[],
                &[],
                &[*ImageMemoryBarrier::builder()
                    .src_access_mask(AccessFlags::MEMORY_READ)
                    .dst_access_mask(AccessFlags::MEMORY_WRITE)
                    .image(img.image)
                    .old_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .subresource_range(img_subresource_range)],
            );
        }

        self.resource_loader.work_buffers.push(work_buffer);
    }
}

impl std::ops::Drop for VulkanState {
    fn drop(&mut self) {}
}

pub struct VulkanPhysicalDeviceState {
    pub device: PhysicalDevice,
    pub properties: PhysicalDeviceProperties,
    pub memory_properties: PhysicalDeviceMemoryProperties,
    pub features: PhysicalDeviceFeatures,
    pub queue_family_id: u32,
}

pub struct VulkanDeviceState {
    pub descriptor_pool: DescriptorPool,
    pub cmd_pool: CommandPool,
    pub queue: Queue,
    pub device: ash::Device,
    pub surface: VulkanSurfaceState,
    pub physical: VulkanPhysicalDeviceState,
}

pub struct VulkanSurfaceKHRState {
    pub ext: ash::extensions::khr::Surface,
    pub surface: SurfaceKHR,
}

pub struct VulkanSurfaceState {
    pub khr: VulkanSurfaceKHRState,
    pub fmt: SurfaceFormatKHR,
    pub present_mode: PresentModeKHR,
    pub transform: SurfaceTransformFlagsKHR,
    pub image_count: u32,
    pub image_size: Extent2D,
}

pub struct VulkanSwapchainState {
    pub ext: ash::extensions::khr::Swapchain,
    pub swapchain: ash::vk::SwapchainKHR,
    pub images: Vec<Image>,
    pub image_views: Vec<ImageView>,
    pub framebuffers: Vec<Framebuffer>,
    pub work_fences: Vec<Fence>,
    pub sem_work_done: Vec<Semaphore>,
    pub sem_img_available: Vec<Semaphore>,
    pub cmd_buffers: Vec<CommandBuffer>,
    pub image_index: u32,
    pub max_frames: u32,
}

impl VulkanSwapchainState {
    pub fn create_swapchain(
        ext: &ash::extensions::khr::Swapchain,
        surface: &VulkanSurfaceState,
        device: &Device,
        renderpass: RenderPass,
        queue: u32,
    ) -> (SwapchainKHR, Vec<Image>, Vec<ImageView>, Vec<Framebuffer>) {
        let swapchain = unsafe {
            ext.create_swapchain(
                &SwapchainCreateInfoKHR::builder()
                    .surface(surface.khr.surface)
                    .min_image_count(surface.image_count)
                    .image_format(surface.fmt.format)
                    .image_color_space(surface.fmt.color_space)
                    .image_extent(surface.image_size)
                    .image_array_layers(1)
                    .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
                    .image_sharing_mode(SharingMode::EXCLUSIVE)
                    .queue_family_indices(&[queue])
                    .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
                    .pre_transform(surface.transform)
                    .present_mode(surface.present_mode),
                None,
            )
            .expect("Failed to create swapchain!")
        };

        let images = unsafe { ext.get_swapchain_images(swapchain) }
            .expect("Failed to get swapchain images ...");

        let image_views = images
            .iter()
            .map(|img| unsafe {
                device
                    .create_image_view(
                        &ImageViewCreateInfo::builder()
                            .image(*img)
                            .view_type(ImageViewType::TYPE_2D)
                            .format(surface.fmt.format)
                            .components(
                                *ComponentMapping::builder()
                                    .r(ComponentSwizzle::IDENTITY)
                                    .g(ComponentSwizzle::IDENTITY)
                                    .b(ComponentSwizzle::IDENTITY)
                                    .a(ComponentSwizzle::IDENTITY),
                            )
                            .subresource_range(
                                *ImageSubresourceRange::builder()
                                    .aspect_mask(ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            ),
                        None,
                    )
                    .expect("Failed to create imageview for swapchain image")
            })
            .collect::<Vec<_>>();

        let framebuffers = image_views
            .iter()
            .map(|&img_view| {
                unsafe {
                    device.create_framebuffer(
                        &FramebufferCreateInfo::builder()
                            .render_pass(renderpass)
                            .attachments(&[img_view])
                            .width(surface.image_size.width)
                            .height(surface.image_size.height)
                            .layers(1),
                        None,
                    )
                }
                .expect("Failed to create framebuffer")
            })
            .collect::<Vec<_>>();

        (swapchain, images, image_views, framebuffers)
    }

    fn create_sync_objects(
        device: &Device,
        frames: u32,
    ) -> (Vec<Fence>, Vec<Semaphore>, Vec<Semaphore>) {
        let work_fences = (0..frames)
            .map(|_| unsafe {
                device
                    .create_fence(
                        &FenceCreateInfo::builder().flags(FenceCreateFlags::SIGNALED),
                        None,
                    )
                    .expect("Failed to create fence ...")
            })
            .collect::<Vec<_>>();

        let sem_work_done = (0..frames)
            .map(|_| unsafe {
                device
                    .create_semaphore(&SemaphoreCreateInfo::builder(), None)
                    .expect("Failed to create semaphore ...")
            })
            .collect::<Vec<_>>();

        let sem_image_available = (0..frames)
            .map(|_| unsafe {
                device
                    .create_semaphore(&SemaphoreCreateInfo::builder(), None)
                    .expect("Failed to create semaphore ...")
            })
            .collect::<Vec<_>>();

        (work_fences, sem_work_done, sem_image_available)
    }

    pub fn new(
        instance: &Instance,
        ds: &VulkanDeviceState,
        renderpass: RenderPass,
    ) -> Option<VulkanSwapchainState> {
        let ext = ash::extensions::khr::Swapchain::new(instance, &ds.device);

        let (swapchain, images, image_views, framebuffers) = Self::create_swapchain(
            &ext,
            &ds.surface,
            &ds.device,
            renderpass,
            ds.physical.queue_family_id,
        );

        let (work_fences, sem_work_done, sem_img_available) =
            Self::create_sync_objects(&ds.device, ds.surface.image_count);

        Some(VulkanSwapchainState {
            ext,
            swapchain,
            images,
            image_views,
            framebuffers,
            work_fences,
            sem_work_done,
            sem_img_available,
            cmd_buffers: unsafe {
                ds.device.allocate_command_buffers(
                    &CommandBufferAllocateInfo::builder()
                        .command_pool(ds.cmd_pool)
                        .command_buffer_count(ds.surface.image_count),
                )
            }
            .expect("Failed to allocate command buffers"),
            image_index: 0,
            max_frames: ds.surface.image_count,
        })
    }

    pub fn handle_suboptimal(&mut self, ds: &VulkanDeviceState, renderpass: RenderPass) {
        unsafe {
            ds.device
                .queue_wait_idle(ds.queue)
                .expect("Failed to wait queue idle");
            ds.device.device_wait_idle().expect("Failed to wait_idle()");

            self.framebuffers.iter().for_each(|fb| {
                ds.device.destroy_framebuffer(*fb, None);
            });

            self.image_views.iter().for_each(|iv| {
                ds.device.destroy_image_view(*iv, None);
            });

            self.ext.destroy_swapchain(self.swapchain, None);
        }

        let (swapchain, images, image_views, framebuffers) = Self::create_swapchain(
            &self.ext,
            &ds.surface,
            &ds.device,
            renderpass,
            ds.physical.queue_family_id,
        );

        self.swapchain = swapchain;
        self.images = images;
        self.image_views = image_views;
        self.framebuffers = framebuffers;
        self.image_index = 0;

        let (fences, work_done, img_avail) = Self::create_sync_objects(&ds.device, self.max_frames);

        self.work_fences = fences;
        self.sem_work_done = work_done;
        self.sem_img_available = img_avail;
    }
}

impl VulkanState {
    unsafe extern "system" fn debug_callback_stub(
        message_severity: ash::vk::DebugUtilsMessageSeverityFlagsEXT,
        _message_types: ash::vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const ash::vk::DebugUtilsMessengerCallbackDataEXT,
        _p_user_data: *mut std::os::raw::c_void,
    ) -> ash::vk::Bool32 {
        if message_severity.intersects(
            DebugUtilsMessageSeverityFlagsEXT::WARNING
                | DebugUtilsMessageSeverityFlagsEXT::ERROR
                | DebugUtilsMessageSeverityFlagsEXT::INFO,
        ) {
            log::debug!(
                "[Vulkan Debug::]\n\t{}",
                std::ffi::CStr::from_ptr((*p_callback_data).p_message)
                    .to_str()
                    .unwrap_or_default()
            );
        }
        ash::vk::FALSE
    }

    fn create_surface(
        entry: &Entry,
        instance: &Instance,
        wsi: WindowSystemIntegration,
    ) -> VulkanSurfaceKHRState {
        let xlib_surface = ash::extensions::khr::XlibSurface::new(entry, instance);

        let khr_surface = unsafe {
            xlib_surface.create_xlib_surface(
                &ash::vk::XlibSurfaceCreateInfoKHR::builder()
                    .dpy(wsi.native_disp as *mut ash::vk::Display)
                    .window(std::mem::transmute::<u64, ash::vk::Window>(wsi.native_win)),
                None,
            )
        }
        .expect("Failed to creale XLIB surface");

        VulkanSurfaceKHRState {
            ext: ash::extensions::khr::Surface::new(entry, instance),
            surface: khr_surface,
        }
    }

    fn pick_device(
        instance: &Instance,
        surface: VulkanSurfaceKHRState,
    ) -> Option<VulkanDeviceState> {
        let phys_devices = unsafe { instance.enumerate_physical_devices() }
            .expect("Failed to query physical devices");

        let mut phys_device: Option<VulkanPhysicalDeviceState> = None;
        let mut surface_state: Option<VulkanSurfaceState> = None;

        for pd in phys_devices {
            let pd_properties = unsafe { instance.get_physical_device_properties(pd) };

            if pd_properties.device_type != PhysicalDeviceType::DISCRETE_GPU
                && pd_properties.device_type != PhysicalDeviceType::INTEGRATED_GPU
            {
                log::info!(
                    "Rejecting device {} (not a GPU device)",
                    pd_properties.device_id
                );
                continue;
            }

            let pd_features = unsafe { instance.get_physical_device_features(pd) };

            if pd_features.multi_draw_indirect == 0 || !pd_features.geometry_shader == 0 {
                log::info!(
                    "Rejecting device {} (no geometry shader and/or MultiDrawIndirect)",
                    pd_properties.device_id
                );
                continue;
            }

            let queue_family_props =
                unsafe { instance.get_physical_device_queue_family_properties(pd) };

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
                    pd_properties.device_id
                );
                continue;
            }

            let queue_id = maybe_queue_id.unwrap() as u32;

            //
            // query surface support
            let device_surface_support = unsafe {
                surface
                    .ext
                    .get_physical_device_surface_support(pd, queue_id, surface.surface)
            }
            .expect("Failed to query device surface support");

            if !device_surface_support {
                log::info!(
                    "Rejecting device {} (does not support surface)",
                    pd_properties.device_id
                );
                continue;
            }

            let surface_caps = unsafe {
                surface
                    .ext
                    .get_physical_device_surface_capabilities(pd, surface.surface)
            }
            .expect("Failed to query device surface caps");

            if !surface_caps
                .supported_usage_flags
                .intersects(ImageUsageFlags::COLOR_ATTACHMENT)
            {
                log::info!(
                    "Rejecting device {} (surface does not support COLOR_ATTACHMENT)",
                    pd_properties.device_id
                );
                continue;
            }

            //
            // query surface formats
            let maybe_surface_format = unsafe {
                surface
                    .ext
                    .get_physical_device_surface_formats(pd, surface.surface)
                    .map(|surface_formats| {
                        surface_formats
                            .iter()
                            .find(|fmt| {
                                fmt.format == Format::B8G8R8A8_UNORM
                                    || fmt.format == Format::R8G8B8A8_UNORM
                            })
                            .copied()
                    })
                    .expect("Failed to query surface format")
            };

            if maybe_surface_format.is_none() {
                log::info!("Rejecting device {} (does not support surface format B8G8R8A8_UNORM/R8G8B8A8_UNORM)", pd_properties.device_id);
                continue;
            }

            //
            // query present mode
            let maybe_present_mode = unsafe {
                surface
                    .ext
                    .get_physical_device_surface_present_modes(pd, surface.surface)
                    .map(|present_modes| {
                        present_modes
                            .iter()
                            .find(|&&mode| {
                                mode == PresentModeKHR::MAILBOX || mode == PresentModeKHR::FIFO
                            })
                            .copied()
                    })
                    .expect("Failed to query present modes")
            };

            if maybe_present_mode.is_none() {
                log::info!(
                    "Rejecting device {} (does not support presentation mode MAILBOX/FIFO)",
                    pd_properties.device_id
                );
                continue;
            }

            let image_count = if surface_caps.max_image_count > 0 {
                (surface_caps.min_image_count + 1).max(surface_caps.max_image_count)
            } else {
                surface_caps.min_image_count + 1
            };

            let image_size = if surface_caps.current_extent.width == 0xFFFFFFFF {
                todo!("Handle this case");
            } else {
                surface_caps.current_extent
            };

            surface_state = Some(VulkanSurfaceState {
                khr: surface,
                image_count,
                image_size,
                transform: surface_caps.current_transform,
                fmt: maybe_surface_format.unwrap(),
                present_mode: maybe_present_mode.unwrap(),
            });

            let memory_properties = unsafe { instance.get_physical_device_memory_properties(pd) };

            phys_device = Some(VulkanPhysicalDeviceState {
                device: pd,
                properties: pd_properties,
                features: pd_features,
                queue_family_id: queue_id,
                memory_properties,
            });

            break;
        }

        if surface_state.is_none() || phys_device.is_none() {
            return None;
        }

        let surface_state = surface_state.unwrap();
        let phys_device = phys_device.unwrap();

        //
        // create logical device
        let enabled_device_extensions = [b"VK_KHR_swapchain\0".as_ptr() as *const c_char];
        let device = unsafe {
            instance.create_device(
                phys_device.device,
                &DeviceCreateInfo::builder()
                    .enabled_extension_names(&enabled_device_extensions)
                    .queue_create_infos(&[DeviceQueueCreateInfo::builder()
                        .queue_family_index(phys_device.queue_family_id)
                        .queue_priorities(&[1f32])
                        .build()]),
                None,
            )
        }
        .expect("Failed to create logical device ...");

        let queue = unsafe { device.get_device_queue(phys_device.queue_family_id, 0) };
        let cmd_pool = unsafe {
            device.create_command_pool(
                &CommandPoolCreateInfo::builder()
                    .queue_family_index(phys_device.queue_family_id)
                    .flags(
                        CommandPoolCreateFlags::TRANSIENT
                            | CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    ),
                None,
            )
        }
        .expect("Failed to create command pool");

        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &DescriptorPoolCreateInfo::builder()
                    .max_sets(4096)
                    .pool_sizes(&[*DescriptorPoolSize::builder()
                        .ty(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .descriptor_count(1024)]),
                None,
            )
        }
        .expect("Failed to create descriptor pool ...");

        Some(VulkanDeviceState {
            device,
            queue,
            physical: phys_device,
            surface: surface_state,
            cmd_pool,
            descriptor_pool,
        })
    }

    pub fn new(wsi: WindowSystemIntegration) -> Option<VulkanState> {
        let entry = Entry::linked();
        let instance_ver = entry
            .try_enumerate_instance_version()
            .expect("Failed to get Vulkan version")?;

        log::info!(
            "Vulkan version: {}.{}.{}",
            ash::vk::api_version_major(instance_ver),
            ash::vk::api_version_minor(instance_ver),
            ash::vk::api_version_patch(instance_ver)
        );

        let mut debug_utils_msg_create_info = DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | DebugUtilsMessageSeverityFlagsEXT::WARNING,
            )
            .message_type(
                DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(VulkanState::debug_callback_stub));

        let app_name_engine = [
            CString::new("Vulkan + Rust adventures").unwrap(),
            CString::new("Epic engine").unwrap(),
        ];

        let app_create_info = ApplicationInfo::builder()
            .api_version(ash::vk::make_api_version(0, 1, 3, 0))
            .application_name(app_name_engine[0].as_c_str())
            .application_version(1)
            .engine_name(app_name_engine[1].as_c_str())
            .engine_version(1);

        let enabled_layers = [b"VK_LAYER_KHRONOS_validation\0".as_ptr() as *const c_char];
        let enabled_instance_extensions = [
            b"VK_KHR_surface\0".as_ptr() as *const c_char,
            b"VK_EXT_debug_utils\0".as_ptr() as *const c_char,
            b"VK_KHR_xlib_surface\0".as_ptr() as *const c_char,
        ];

        unsafe {
            entry
                .enumerate_instance_extension_properties(None)
                .map(|instance_exts| {
                    instance_exts.iter().for_each(|ext| {
                        let ext_name = std::ffi::CStr::from_ptr(ext.extension_name.as_ptr())
                            .to_str()
                            .unwrap();
                        log::info!("[instance extension :: {}]", ext_name);
                    });
                })
                .expect("Failed to enumerate instance extensions ...");
        }

        let instance = unsafe {
            entry.create_instance(
                &InstanceCreateInfo::builder()
                    .application_info(&app_create_info)
                    .push_next(&mut debug_utils_msg_create_info)
                    .enabled_layer_names(&enabled_layers)
                    .enabled_extension_names(&enabled_instance_extensions),
                None,
            )
        }
        .expect("Failed to create instance");

        let (dbg, msgr) = unsafe {
            let dbg = DebugUtils::new(&entry, &instance);
            let msgr = dbg
                .create_debug_utils_messenger(&debug_utils_msg_create_info, None)
                .expect("Failed to create debug messenger");
            (dbg, msgr)
        };

        let surface_state = Self::create_surface(&entry, &instance, wsi);

        let device_state =
            Self::pick_device(&instance, surface_state).expect("Faile to pick device");

        let renderpass = Self::create_renderpass(&device_state);

        let swapchain_state =
            VulkanSwapchainState::new(&instance, &device_state, renderpass).unwrap();

        let resource_loader = ResourceLoadingState::new(&device_state);

        Some(VulkanState {
            dbg,
            msgr,
            entry,
            instance,
            ds: device_state,
            swapchain: swapchain_state,
            renderpass,
            resource_loader,
        })
    }

    fn create_renderpass(ds: &VulkanDeviceState) -> RenderPass {
        let attachment_descriptions = [AttachmentDescription::builder()
            .format(ds.surface.fmt.format)
            .samples(SampleCountFlags::TYPE_1)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::STORE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::PRESENT_SRC_KHR)
            .build()];

        let attachment_refs = [AttachmentReference::builder()
            .attachment(0)
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()];

        unsafe {
            ds.device.create_render_pass(
                &RenderPassCreateInfo::builder()
                    .attachments(&attachment_descriptions)
                    .subpasses(&[SubpassDescription::builder()
                        .color_attachments(&attachment_refs)
                        .pipeline_bind_point(PipelineBindPoint::GRAPHICS)
                        .build()]),
                None,
            )
        }
        .expect("Failed to create renderpass")
    }

    pub fn begin_rendering(&mut self, fb_size: Extent2D) -> FrameRenderContext {
        //
        // wait for previous submittted work
        unsafe {
            self.ds
                .device
                .wait_for_fences(
                    &[self.swapchain.work_fences[self.swapchain.image_index as usize]],
                    true,
                    u64::MAX,
                )
                .expect("Failed to wait for submits");
        }

        //
        // acquire next image

        let (swapchain_available_img_index, suboptimal) = unsafe {
            self.swapchain.ext.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                self.swapchain.sem_img_available[self.swapchain.image_index as usize],
                Fence::null(),
            )
        }
        .expect("Acquire image error ...");

        if suboptimal {
            todo!("handle swapchain suboptimal!");
        }

        unsafe {
            self.ds
                .device
                .reset_fences(&[self.swapchain.work_fences[self.swapchain.image_index as usize]])
                .expect("Failed to reset fence ...");
        }

        //
        // begin command buffer + renderpass

        unsafe {
            self.ds
                .device
                .reset_command_buffer(
                    self.swapchain.cmd_buffers[self.swapchain.image_index as usize],
                    CommandBufferResetFlags::empty(),
                )
                .expect("Failed to reset command buffer");

            self.ds
                .device
                .begin_command_buffer(
                    self.swapchain.cmd_buffers[self.swapchain.image_index as usize],
                    &CommandBufferBeginInfo::builder()
                        .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("Failed to begin command buffer ...");
        }

        FrameRenderContext {
            cmd_buff: self.swapchain.cmd_buffers[self.swapchain.image_index as usize],
            framebuffer: self.swapchain.framebuffers[swapchain_available_img_index as usize],
            fb_size,
            current_frame_id: self.swapchain.image_index,
        }
    }

    pub fn end_rendering(&mut self) {
        //
        // end command buffer + renderpass
        unsafe {
            self.ds
                .device
                .end_command_buffer(self.swapchain.cmd_buffers[self.swapchain.image_index as usize])
                .expect("Failed to end command buffer");
        }

        //
        // submit
        unsafe {
            self.ds
                .device
                .queue_submit(
                    self.ds.queue,
                    &[*SubmitInfo::builder()
                        .command_buffers(&[
                            self.swapchain.cmd_buffers[self.swapchain.image_index as usize]
                        ])
                        .wait_semaphores(&[
                            self.swapchain.sem_img_available[self.swapchain.image_index as usize]
                        ])
                        .wait_dst_stage_mask(&[PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                        .signal_semaphores(&[
                            self.swapchain.sem_work_done[self.swapchain.image_index as usize]
                        ])],
                    self.swapchain.work_fences[self.swapchain.image_index as usize],
                )
                .expect("Failed to submit work");

            match self.swapchain.ext.queue_present(
                self.ds.queue,
                &PresentInfoKHR::builder()
                    .image_indices(&[self.swapchain.image_index])
                    .swapchains(&[self.swapchain.swapchain])
                    .wait_semaphores(&[
                        self.swapchain.sem_work_done[self.swapchain.image_index as usize]
                    ]),
            ) {
                Err(e) => {
                    if e == ash::vk::Result::ERROR_OUT_OF_DATE_KHR {
                        log::info!("Swapchain out of date, recreating ...");
                        self.handle_surface_size_changed();
                        self.swapchain.handle_suboptimal(&self.ds, self.renderpass);
                    } else {
                        todo!("Handle this ...");
                    }
                }
                Ok(suboptimal) => {
                    if suboptimal {
                        log::info!("Swapchain suboptimal, recreating ...");
                        self.handle_surface_size_changed();
                        self.swapchain.handle_suboptimal(&self.ds, self.renderpass);
                    } else {
                        self.swapchain.image_index =
                            (self.swapchain.image_index + 1) % self.swapchain.max_frames;
                    }
                }
            };
        }
    }

    pub fn handle_surface_size_changed(&mut self) {
        let surface_caps = unsafe {
            self.ds
                .surface
                .khr
                .ext
                .get_physical_device_surface_capabilities(
                    self.ds.physical.device,
                    self.ds.surface.khr.surface,
                )
        }
        .expect("Failed to query surface caps");

        log::info!("Surface extent {:?}", surface_caps.current_extent);

        assert_ne!(surface_caps.current_extent.width, 0xFFFFFFFF);
        self.ds.surface.image_size = surface_caps.current_extent;
    }
}

pub struct FrameRenderContext {
    pub cmd_buff: CommandBuffer,
    pub framebuffer: Framebuffer,
    pub fb_size: Extent2D,
    pub current_frame_id: u32,
}
