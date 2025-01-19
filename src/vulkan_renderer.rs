use std::{
    ffi::{c_char, c_void, CString},
    mem::size_of,
};

use ash::{
    extensions::ext::DebugUtils,
    vk::{
        AccessFlags, AccessFlags2, ApplicationInfo, AttachmentDescription, AttachmentLoadOp,
        AttachmentReference, AttachmentStoreOp, BlendFactor, BlendOp, Buffer, BufferCopy,
        BufferCreateFlags, BufferCreateInfo, BufferImageCopy, BufferMemoryRequirementsInfo2,
        BufferUsageFlags, ClearColorValue, ClearDepthStencilValue, ClearValue, ColorComponentFlags,
        CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel,
        CommandBufferResetFlags, CommandBufferUsageFlags, CommandPool, CommandPoolCreateFlags,
        CommandPoolCreateInfo, CompareOp, ComponentMapping, ComponentSwizzle,
        CompositeAlphaFlagsKHR, CullModeFlags, DebugUtilsLabelEXT,
        DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
        DebugUtilsMessengerCreateInfoEXT, DebugUtilsMessengerEXT, DependencyFlags, DependencyInfo,
        DescriptorBindingFlags, DescriptorBufferInfo, DescriptorImageInfo, DescriptorPool,
        DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSet, DescriptorSetAllocateInfo,
        DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutBindingFlagsCreateInfo,
        DescriptorSetLayoutCreateInfo, DescriptorType, DeviceCreateInfo, DeviceMemory,
        DeviceQueueCreateInfo, DeviceSize, DynamicState, Extent2D, Fence, FenceCreateFlags,
        FenceCreateInfo, Format, Framebuffer, FramebufferCreateInfo, FrontFace,
        GraphicsPipelineCreateInfo, Image, ImageAspectFlags, ImageCreateFlags, ImageCreateInfo,
        ImageLayout, ImageMemoryBarrier, ImageMemoryBarrier2, ImageSubresourceLayers,
        ImageSubresourceRange, ImageUsageFlags, ImageView, ImageViewCreateInfo, ImageViewType,
        InstanceCreateInfo, MappedMemoryRange, MemoryAllocateInfo, MemoryMapFlags,
        MemoryPropertyFlags, MemoryRequirements, MemoryRequirements2, Offset2D, PhysicalDevice,
        PhysicalDeviceFeatures, PhysicalDeviceFeatures2, PhysicalDeviceMemoryProperties,
        PhysicalDeviceProperties2, PhysicalDeviceType, PhysicalDeviceVulkan11Features,
        PhysicalDeviceVulkan11Properties, PhysicalDeviceVulkan12Features,
        PhysicalDeviceVulkan12Properties, PhysicalDeviceVulkan13Features,
        PhysicalDeviceVulkan13Properties, Pipeline, PipelineBindPoint, PipelineCache,
        PipelineColorBlendStateCreateInfo, PipelineDepthStencilStateCreateInfo,
        PipelineDynamicStateCreateInfo, PipelineLayout, PipelineLayoutCreateInfo,
        PipelineMultisampleStateCreateInfo, PipelineRasterizationStateCreateInfo,
        PipelineRenderingCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags,
        PipelineStageFlags2, PipelineVertexInputStateCreateInfo, PolygonMode, PresentInfoKHR,
        PresentModeKHR, PushConstantRange, Queue, Rect2D, RenderPassBeginInfo,
        RenderPassCreateInfo, RenderingAttachmentInfo, RenderingInfo, SampleCountFlags, Sampler,
        SamplerCreateInfo, Semaphore, SemaphoreCreateInfo, ShaderModule, ShaderStageFlags,
        SharingMode, SubmitInfo, SubpassContents, SubpassDescription, SurfaceFormatKHR, SurfaceKHR,
        SwapchainCreateInfoKHR, SwapchainKHR, VertexInputAttributeDescription, VertexInputRate,
        WriteDescriptorSet, WHOLE_SIZE,
    },
    Device, Entry, Instance,
};

use ash::{
    prelude::*,
    vk::{Extent3D, ImageTiling, ImageType},
};
use smallvec::SmallVec;

use crate::{
    shader::ShaderSource,
    spin_mutex::{self, SpinMutex},
};

trait VkObjectType {
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

#[derive(Copy, Clone, Debug, thiserror::Error)]
pub enum VulkanSystemError {
    #[error("vulkan api error")]
    VulkanResult(#[from] ash::vk::Result),
}

pub struct UniqueImage {
    device: *const Device,
    pub image: Image,
    pub memory: DeviceMemory,
}

impl UniqueImage {
    pub fn from_bytes(
        vks: &mut VulkanRenderer,
        create_info: ImageCreateInfo,
        pixels: &[u8],
    ) -> UniqueImage {
        let image = unsafe { vks.device_state.device.create_image(&create_info, None) }
            .expect("Failed to create image");

        let memory_req = unsafe { vks.device_state.device.get_image_memory_requirements(image) };

        let image_memory = unsafe {
            vks.device_state.device.allocate_memory(
                &MemoryAllocateInfo::builder()
                    .allocation_size(memory_req.size)
                    .memory_type_index(choose_memory_heap(
                        &memory_req,
                        MemoryPropertyFlags::DEVICE_LOCAL,
                        vks.memory_properties(),
                    )),
                None,
            )
        }
        .expect("Failed to allocate memory");

        unsafe {
            vks.device_state
                .device
                .bind_image_memory(image, image_memory, 0)
        }
        .expect(&format!(
            "Failed to bind memory @ {:?} for image {:?}",
            image, image_memory
        ));

        let img = UniqueImage {
            device: &vks.device_state.device as *const _,
            image,
            memory: image_memory,
        };

        todo!("Fix this");

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
        vks: &VulkanRenderer,
        img: &UniqueImage,
        img_view_create_info: ImageViewCreateInfo,
    ) -> UniqueImageView {
        let view = unsafe {
            vks.device_state
                .device
                .create_image_view(&img_view_create_info, None)
        }
        .expect("Failed to create image view");

        UniqueImageView {
            device: &vks.device_state.device as *const _,
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
    pub fn new(vks: &VulkanRenderer, create_info: SamplerCreateInfo) -> UniqueSampler {
        let handle = unsafe { vks.device_state.device.create_sampler(&create_info, None) }
            .expect("Failed to create sampler");

        UniqueSampler {
            device: &vks.device_state.device as *const _,
            handle,
        }
    }
}

enum PipelineData {
    Owned {
        layout: PipelineLayout,
        descriptor_set_layout: Vec<DescriptorSetLayout>,
    },
    Reference {
        layout: PipelineLayout,
    },
}

impl PipelineData {
    fn layout(&self) -> PipelineLayout {
        match self {
            PipelineData::Reference { layout } => *layout,
            PipelineData::Owned { layout, .. } => *layout,
        }
    }
}

pub struct UniquePipeline {
    device: *const Device,
    handle: Pipeline,
    data: PipelineData,
}

impl std::ops::Drop for UniquePipeline {
    fn drop(&mut self) {
        match &self.data {
            PipelineData::Owned {
                layout,
                descriptor_set_layout,
            } => {
                descriptor_set_layout.iter().for_each(|&ds_layout| unsafe {
                    (*self.device).destroy_descriptor_set_layout(ds_layout, None);
                });

                unsafe {
                    (*self.device).destroy_pipeline_layout(*layout, None);
                }
            }

            PipelineData::Reference { .. } => {}
        }

        unsafe {
            (*self.device).destroy_pipeline(self.handle, None);
        }
    }
}

impl UniquePipeline {
    pub fn handle(&self) -> ash::vk::Pipeline {
        self.handle
    }

    pub fn layout(&self) -> ash::vk::PipelineLayout {
        self.data.layout()
    }
}

pub fn choose_memory_heap(
    memory_req: &MemoryRequirements,
    required_flags: MemoryPropertyFlags,
    memory_properties: &PhysicalDeviceMemoryProperties,
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

pub struct UniqueShaderModule {
    pub(crate) device: *const ash::Device,
    pub(crate) module: ShaderModule,
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

pub struct VulkanBufferCreateInfo<'a> {
    pub name_tag: Option<&'a str>,
    pub work_package: Option<ash::vk::CommandBuffer>,
    pub usage: BufferUsageFlags,
    pub memory_properties: MemoryPropertyFlags,
    pub slabs: usize,
    pub bytes: usize,
    pub initial_data: &'a [&'a [u8]],
}

pub struct VulkanBuffer {
    device: *const Device,
    pub buffer: Buffer,
    pub memory: DeviceMemory,
    pub slabs: usize,
    pub aligned_slab_size: usize,
}

impl VulkanBuffer {
    pub fn size_bytes(&self) -> usize {
        self.aligned_slab_size * self.slabs
    }

    pub fn create(
        renderer: &mut VulkanRenderer,
        create_info: &VulkanBufferCreateInfo,
    ) -> std::result::Result<VulkanBuffer, GraphicsError> {
        let memory_host_access: MemoryPropertyFlags = MemoryPropertyFlags::HOST_VISIBLE
            | MemoryPropertyFlags::HOST_COHERENT
            | MemoryPropertyFlags::HOST_CACHED;

        let alignment = if create_info.memory_properties.intersects(memory_host_access) {
            renderer.limits().non_coherent_atom_size
        } else {
            //
            // no host access, must have initial data
            if create_info.initial_data.is_empty() {
                Err(GraphicsError::Generic(
                    "Buffer has no host access flags but is missing initial data".into(),
                ))?;
            }

            if create_info
                .usage
                .intersects(BufferUsageFlags::UNIFORM_BUFFER)
            {
                renderer.limits().min_uniform_buffer_offset_alignment
            } else if create_info
                .usage
                .intersects(BufferUsageFlags::STORAGE_BUFFER)
            {
                renderer.limits().min_storage_buffer_offset_alignment
            } else {
                renderer.limits().non_coherent_atom_size
            }
        };

        let aligned_slab_size = round_up(create_info.bytes, alignment as usize);
        let aligned_allocation_size = aligned_slab_size * create_info.slabs as usize;
        let initial_data_size = create_info
            .initial_data
            .iter()
            .map(|data_slice| data_slice.len())
            .sum::<usize>();

        assert!(initial_data_size <= aligned_allocation_size);

        //
        // when creating an immutable buffer add TRANSFER_DST to usage flags
        let usage_flags = create_info.usage
            | (if create_info.memory_properties.intersects(memory_host_access) {
                BufferUsageFlags::empty()
            } else {
                BufferUsageFlags::TRANSFER_DST
            });

        let device = renderer.device_raw();
        let buffer = unsafe {
            (*device).create_buffer(
                &BufferCreateInfo::builder()
                    .size(aligned_allocation_size as u64)
                    .usage(usage_flags)
                    .sharing_mode(SharingMode::EXCLUSIVE),
                None,
            )
        }?;

        scopeguard::defer_on_unwind! {
            unsafe {
                (*device).destroy_buffer(buffer, None);
            }
        }

        create_info
            .name_tag
            .map(|name_tag| renderer.debug_set_object_name(buffer, name_tag));

        let buffer_memory = unsafe {
            let mut memory_requirements = *MemoryRequirements2::builder();
            (*device).get_buffer_memory_requirements2(
                &BufferMemoryRequirementsInfo2::builder().buffer(buffer),
                &mut memory_requirements,
            );

            (*device).allocate_memory(
                &MemoryAllocateInfo::builder()
                    .allocation_size(memory_requirements.memory_requirements.size)
                    .memory_type_index(choose_memory_heap(
                        &memory_requirements.memory_requirements,
                        create_info.memory_properties,
                        renderer.memory_properties(),
                    )),
                None,
            )
        }?;
        scopeguard::defer_on_unwind! {
            unsafe {
                (*device).free_memory(buffer_memory, None);
            }
        }

        unsafe { (*device).bind_buffer_memory(buffer, buffer_memory, 0) }?;

        if create_info.memory_properties.intersects(memory_host_access) {
            //
            // not immutable so just copy everything there is to copy
            if initial_data_size != 0 {
                UniqueBufferMapping::map_memory(
                    renderer.logical(),
                    buffer_memory,
                    0,
                    aligned_allocation_size,
                )
                .map(|buffer_mapping| {
                    let _ =
                        create_info
                            .initial_data
                            .iter()
                            .fold(0isize, |copy_offset, data| unsafe {
                                std::ptr::copy_nonoverlapping(
                                    data.as_ptr(),
                                    (buffer_mapping.mapped_memory as *mut u8).offset(copy_offset),
                                    data.len(),
                                );
                                copy_offset + data.len() as isize
                            });
                })?;
            }
        } else {
            //
            // immutable buffer, create a staging buffer to use as a copy source then issue a copy command
            assert!(create_info.work_package.is_some());

            let (staging_ptr, staging_buffer, staging_offset) =
                renderer.reserve_staging_memory(aligned_allocation_size);

            let mut copy_offset = 0isize;
            let copy_regions = create_info
                .initial_data
                .iter()
                .map(|copy_slice| {
                    let copy_buffer = *BufferCopy::builder()
                        .src_offset(staging_offset as DeviceSize + copy_offset as DeviceSize)
                        .dst_offset(copy_offset as DeviceSize)
                        .size(copy_slice.len() as DeviceSize);
                    copy_offset += copy_slice.len() as isize;

                    copy_buffer
                })
                .collect::<smallvec::SmallVec<[BufferCopy; 4]>>();

            //
            // copy data from host to GPU staging buffer
            let _ = create_info
                .initial_data
                .iter()
                .fold(0isize, |copy_offset, data| unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        staging_ptr.byte_offset(staging_offset as isize + copy_offset),
                        data.len(),
                    );
                    copy_offset + data.len() as isize
                });

            //
            // copy between staging buffer and destination buffer
            unsafe {
                (*device).cmd_copy_buffer(
                    create_info.work_package.unwrap(),
                    staging_buffer,
                    buffer,
                    &copy_regions,
                );
            }
        }

        let buffer_name = create_info.name_tag.unwrap_or_else(|| "anonymous");

        log::info!(
            "New buffer [[{buffer_name}]] @ {buffer:p} <=> {buffer_memory:p}, alignment {alignment}, 
            aligned slab size {aligned_slab_size}, aligned allocation size {aligned_allocation_size}"
        );

        Ok(VulkanBuffer {
            device,
            buffer,
            memory: buffer_memory,
            slabs: create_info.slabs,
            aligned_slab_size,
        })
    }

    pub fn map_slab<'a>(
        &self,
        renderer: &'a VulkanRenderer,
        slab: u32,
    ) -> std::result::Result<UniqueBufferMapping<'a>, GraphicsError> {
        assert!(slab < self.slabs as u32);
        UniqueBufferMapping::map_memory(
            renderer.logical(),
            self.memory,
            self.aligned_slab_size * slab as usize,
            self.aligned_slab_size,
        )
    }
}

// pub struct UniqueBuffer {
//     device: *const Device,
//     pub handle: Buffer,
//     pub memory: DeviceMemory,
//     pub aligned_item_size: usize,
// }
//
// impl std::ops::Drop for UniqueBuffer {
//     fn drop(&mut self) {
//         log::debug!(
//             "Dropping buffer and memory {:?} -> {:?}",
//             self.handle,
//             self.memory
//         );
//         unsafe {
//             (*self.device).destroy_buffer(self.handle, None);
//             (*self.device).free_memory(self.memory, None);
//         }
//     }
// }
//
// impl UniqueBuffer {
//     // pub fn with_capacity(
//     //     ds: &VulkanRenderer,
//     //     usage: BufferUsageFlags,
//     //     memory_flags: MemoryPropertyFlags,
//     //     items: usize,
//     //     item_size: usize,
//     // ) -> Self {
//     //     let align_size = if memory_flags.intersects(
//     //         MemoryPropertyFlags::HOST_VISIBLE
//     //             | MemoryPropertyFlags::HOST_COHERENT
//     //             | MemoryPropertyFlags::HOST_CACHED,
//     //     ) {
//     //         ds.device_state
//     //             .physical
//     //             .properties
//     //             .base
//     //             .properties
//     //             .limits
//     //             .non_coherent_atom_size
//     //     } else {
//     //         if usage.intersects(BufferUsageFlags::UNIFORM_BUFFER) {
//     //             ds.device_state
//     //                 .physical
//     //                 .properties
//     //                 .base
//     //                 .properties
//     //                 .limits
//     //                 .min_uniform_buffer_offset_alignment
//     //         } else if usage.intersects(
//     //             BufferUsageFlags::UNIFORM_TEXEL_BUFFER | BufferUsageFlags::STORAGE_TEXEL_BUFFER,
//     //         ) {
//     //             ds.device_state
//     //                 .physical
//     //                 .properties
//     //                 .base
//     //                 .properties
//     //                 .limits
//     //                 .min_texel_buffer_offset_alignment
//     //         } else if usage.intersects(BufferUsageFlags::STORAGE_BUFFER) {
//     //             ds.device_state
//     //                 .physical
//     //                 .properties
//     //                 .base
//     //                 .properties
//     //                 .limits
//     //                 .min_storage_buffer_offset_alignment
//     //         } else {
//     //             ds.device_state
//     //                 .physical
//     //                 .properties
//     //                 .base
//     //                 .properties
//     //                 .limits
//     //                 .non_coherent_atom_size
//     //         }
//     //     } as usize;
//     //
//     //     let aligned_item_size = round_up(item_size, align_size);
//     //     let size = aligned_item_size * items;
//     //
//     //     let buffer = unsafe {
//     //         ds.device_state.device.create_buffer(
//     //             &BufferCreateInfo::builder()
//     //                 .size(size as DeviceSize)
//     //                 .usage(usage)
//     //                 .sharing_mode(SharingMode::EXCLUSIVE)
//     //                 .queue_family_indices(&[]),
//     //             None,
//     //         )
//     //     }?;
//     //
//     //     let memory_req = unsafe {
//     //         ds.device_state
//     //             .device
//     //             .get_buffer_memory_requirements(buffer)
//     //     };
//     //     let mem_heap = choose_memory_heap(&memory_req, memory_flags, ds.memory_properties());
//     //
//     //     let buffer_memory = unsafe {
//     //         ds.device_state.device.allocate_memory(
//     //             &MemoryAllocateInfo::builder()
//     //                 .allocation_size(memory_req.size)
//     //                 .memory_type_index(mem_heap),
//     //             None,
//     //         )
//     //     }?;
//     //
//     //     unsafe {
//     //         ds.device_state
//     //             .device
//     //             .bind_buffer_memory(buffer, buffer_memory, 0)
//     //             .expect("Failed to bind memory for buffer");
//     //     }
//     //
//     //     log::debug!(
//     //         "Create buffer and memory object {:?} -> {:?} -> {}",
//     //         buffer,
//     //         buffer_memory,
//     //         memory_req.size
//     //     );
//     //
//     //     Self {
//     //         device: &ds.device_state.device as *const _,
//     //         handle: buffer,
//     //         memory: buffer_memory,
//     //         aligned_item_size,
//     //     }
//     // }
//     //
//     // pub fn new<T: Sized>(
//     //     ds: &VulkanRenderer,
//     //     usage: BufferUsageFlags,
//     //     memory_flags: MemoryPropertyFlags,
//     //     items: usize,
//     // ) -> Self {
//     //     Self::with_capacity(ds, usage, memory_flags, items, std::mem::size_of::<T>())
//     // }
// }

fn round_up(num_to_round: usize, multiple: usize) -> usize {
    assert_ne!(multiple, 0);
    if num_to_round == 0 {
        0
    } else {
        ((num_to_round - 1) / multiple + 1) * multiple
    }
}

pub struct UniqueBufferMapping<'a> {
    buffer_mem: DeviceMemory,
    mapped_memory: *mut c_void,
    device: &'a Device,
    offset: usize,
    range_start: usize,
    range_size: usize,
}

impl<'a> UniqueBufferMapping<'a> {
    pub fn map_memory(
        device: &'a ash::Device,
        device_memory: DeviceMemory,
        offset: usize,
        map_size: usize,
    ) -> std::result::Result<Self, GraphicsError> {
        let mapped_memory = unsafe {
            device.map_memory(
                device_memory,
                offset as u64,
                map_size as u64,
                MemoryMapFlags::empty(),
            )
        }?;

        Ok(Self {
            buffer_mem: device_memory,
            mapped_memory,
            device,
            offset,
            range_start: offset,
            range_size: map_size,
        })
    }

    pub fn new(
        buf: &UniqueBuffer,
        ds: &'a VulkanDeviceState,
        offset: Option<usize>,
        size: Option<usize>,
    ) -> Self {
        let round_down = |x: usize, m: usize| (x / m) * m;

        let offset = offset.unwrap_or_default();
        let range_start = round_down(offset, buf.aligned_item_size);

        assert!(offset >= range_start);

        let range_size = size
            .map(|s| round_up(s, buf.aligned_item_size))
            .unwrap_or(ash::vk::WHOLE_SIZE as usize);

        let mapped_memory = unsafe {
            ds.device.map_memory(
                buf.memory,
                range_start as u64,
                range_size as u64,
                MemoryMapFlags::empty(),
            )
        }
        .expect("Failed to map memory");

        Self {
            buffer_mem: buf.memory,
            mapped_memory,
            device: &ds.device,
            offset: offset - range_start,
            range_start,
            range_size,
        }
    }

    pub fn write_data<T: Sized + Copy>(&self, data: &[T]) {
        unsafe {
            let dst = (self.mapped_memory as *mut u8).offset(self.offset as isize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst as *mut T, data.len());
        }
    }

    pub fn write_data_with_offset<T: Sized + Copy>(&mut self, data: &[T], offset: isize) {
        unsafe {
            let dst = (self.mapped_memory as *mut u8).offset(self.offset as isize);
            let dst = (dst as *mut T).offset(offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst as *mut T, data.len());
        }
    }
}

impl<'a> std::ops::Drop for UniqueBufferMapping<'a> {
    fn drop(&mut self) {
        unsafe {
            self.device
                .flush_mapped_memory_ranges(&[*MappedMemoryRange::builder()
                    .memory(self.buffer_mem)
                    .offset(self.range_start as DeviceSize)
                    .size(self.range_size as DeviceSize)])
                .expect("Failed to flush mapped memory ...");

            self.device.unmap_memory(self.buffer_mem);
        }
    }
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
    mapped_memory: *mut c_void,
    free_ptr: std::sync::atomic::AtomicUsize,
}

impl StagingSystem {
    fn create(
        device: &ash::Device,
        memory_properties: &PhysicalDeviceMemoryProperties,
    ) -> Result<StagingSystem, GraphicsError> {
        const STAGING_BUFFER_SIZE: u32 = 512 * 1024 * 2014;

        let staging_buffer = unsafe {
            device.create_buffer(
                &BufferCreateInfo::builder()
                    .usage(BufferUsageFlags::TRANSFER_SRC)
                    .size(STAGING_BUFFER_SIZE as u64)
                    .sharing_mode(SharingMode::EXCLUSIVE),
                None,
            )
        }?;

        let mem_req = unsafe { device.get_buffer_memory_requirements(staging_buffer) };
        let staging_memory = unsafe {
            device.allocate_memory(
                &MemoryAllocateInfo::builder()
                    .allocation_size(mem_req.size)
                    .memory_type_index(choose_memory_heap(
                        &mem_req,
                        MemoryPropertyFlags::HOST_COHERENT,
                        memory_properties,
                    )),
                None,
            )
        }?;

        unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0) }?;

        let mapped_memory = unsafe {
            device.map_memory(
                staging_memory,
                0,
                STAGING_BUFFER_SIZE as ash::vk::DeviceSize,
                MemoryMapFlags::empty(),
            )
        }?;

        Ok(StagingSystem {
            staging_buffer,
            staging_memory,
            mapped_memory,
            free_ptr: std::sync::atomic::AtomicUsize::new(0),
        })
    }
}

pub struct VulkanRenderer {
    staging_sys: StagingSystem,
    pub renderstate: RenderState,
    pub swapchain: VulkanSwapchainState,
    pub device_state: VulkanDeviceState,
    pub msgr: DebugUtilsMessengerEXT,
    pub dbg: DebugUtils,
    pub instance: Instance,
    pub entry: Entry,
}

unsafe impl Send for VulkanRenderer {}
unsafe impl Sync for VulkanRenderer {}

struct QueueSubmitWaitToken {
    cmd_buffer: ash::vk::CommandBuffer,
    wait_fence: ash::vk::Fence,
}

impl VulkanRenderer {
    pub fn create_queue_job(
        &self,
        qtype: QueueType,
    ) -> Result<ash::vk::CommandBuffer, GraphicsError> {
        let (_, _, cmd_pool, pool_lock, _) = self.device_state.queue_data(qtype);
        let _ = pool_lock.lock();
        unsafe {
            let cmd_buffers = self.device_state.device.allocate_command_buffers(
                &CommandBufferAllocateInfo::builder()
                    .level(CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1)
                    .command_pool(cmd_pool),
            )?;

            self.device_state.device.begin_command_buffer(
                cmd_buffers[0],
                &CommandBufferBeginInfo::builder().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            Ok(cmd_buffers[0])
        }
    }

    // pub fn submit_queue_job(
    //     &self,
    //     cmd_buf: CommandBuffer,
    //     qtype: QueueType,
    // ) -> Result<QueueSubmitWaitToken, GraphicsError> {
    //     let (_, queue, cmd_pool, pool_lock, queue_lock) = self.device_state.queue_data(qtype);
    //
    //     let wait_fence = unsafe {
    //         self.device_state.device.end_command_buffer(cmd_buf)?;
    //         self.device_state
    //             .device
    //             .create_fence(&FenceCreateInfo::builder(), None)
    //     }?;
    //
    //     scopeguard::defer_on_unwind! {
    //         unsafe {
    //             self.device_state.device.destroy_fence(wait_fence, None);
    //             let _ = pool_lock.lock();
    //             self.device_state.device.free_command_buffers(cmd_pool, &[cmd_buf]);
    //         }
    //     }
    //
    //     unsafe {
    //         self.device_state.device.queue_submit(queue, submits, fence)
    //     }
    // }

    pub fn reserve_staging_memory(&self, bytes: usize) -> (*mut u8, ash::vk::Buffer, usize) {
        let byte_offset = self
            .staging_sys
            .free_ptr
            .fetch_add(bytes, std::sync::atomic::Ordering::Acquire);

        (
            unsafe {
                (self.staging_sys.mapped_memory as *mut u8).byte_offset(byte_offset as isize)
            },
            self.staging_sys.staging_buffer,
            byte_offset,
        )
    }

    pub fn setup(&self) -> (u32,) {
        (self.swapchain.max_frames,)
    }
    pub fn limits(&self) -> &ash::vk::PhysicalDeviceLimits {
        &self.device_state.physical.properties.base.properties.limits
    }

    pub fn memory_properties(&self) -> &PhysicalDeviceMemoryProperties {
        &self.device_state.physical.memory_properties
    }

    pub fn features(&self) -> &PhysicalDeviceFeatures {
        &self.device_state.physical.features.base.features
    }

    pub fn empty_descriptor_set_layout(&self) -> DescriptorSetLayout {
        self.device_state.empty_descriptor_layout
    }

    pub fn pipeline_cache(&self) -> PipelineCache {
        self.device_state.pipeline_cache
    }

    pub fn device_raw(&self) -> *const ash::Device {
        &self.device_state.device as *const _
    }

    pub fn wait_all_idle(&mut self) {
        unsafe {
            // self.device_state
            //     .device
            //     .queue_wait_idle(self.device_state.queue)
            //     .expect("Failed to wait for idle queue");
            self.device_state
                .device
                .device_wait_idle()
                .expect("Failed to wait for device idle");
        }
    }

    // pub fn copy_pixels_to_image(
    //     &mut self,
    //     img: &UniqueImage,
    //     pixels: &[u8],
    //     image_info: &ImageCreateInfo,
    // ) {
    //     let work_buffer = UniqueBuffer::new::<u8>(
    //         self,
    //         BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_SRC,
    //         MemoryPropertyFlags::HOST_VISIBLE,
    //         pixels.len(),
    //     );
    //
    //     UniqueBufferMapping::new(&work_buffer, &self.device_state, None, None).write_data(pixels);
    //
    //     let img_subresource_range = *ImageSubresourceRange::builder()
    //         .aspect_mask(ImageAspectFlags::COLOR)
    //         .layer_count(image_info.array_layers)
    //         .base_array_layer(0)
    //         .level_count(image_info.mip_levels)
    //         .base_mip_level(0);
    //
    //     //
    //     // transition image layout from undefined -> transfer src
    //     unsafe {
    //         self.device_state.device.cmd_pipeline_barrier(
    //             self.resource_loader.cmd_buf,
    //             PipelineStageFlags::TOP_OF_PIPE,
    //             PipelineStageFlags::TRANSFER,
    //             DependencyFlags::empty(),
    //             &[],
    //             &[],
    //             &[*ImageMemoryBarrier::builder()
    //                 .src_access_mask(AccessFlags::NONE)
    //                 .dst_access_mask(AccessFlags::TRANSFER_WRITE)
    //                 .image(img.image)
    //                 .old_layout(ImageLayout::UNDEFINED)
    //                 .new_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
    //                 .subresource_range(img_subresource_range)],
    //         );
    //     }
    //
    //     //
    //     // copy pixels
    //     unsafe {
    //         self.device_state.device.cmd_copy_buffer_to_image(
    //             self.resource_loader.cmd_buf,
    //             work_buffer.handle,
    //             img.image,
    //             ImageLayout::TRANSFER_DST_OPTIMAL,
    //             &[*BufferImageCopy::builder()
    //                 .buffer_offset(0)
    //                 .image_subresource(
    //                     *ImageSubresourceLayers::builder()
    //                         .aspect_mask(ImageAspectFlags::COLOR)
    //                         .base_array_layer(0)
    //                         .layer_count(image_info.array_layers)
    //                         .mip_level(0),
    //                 )
    //                 .image_extent(image_info.extent)],
    //         );
    //     }
    //
    //     //
    //     // transition layout from transfer -> shader readonly optimal
    //     unsafe {
    //         self.device_state.device.cmd_pipeline_barrier(
    //             self.resource_loader.cmd_buf,
    //             PipelineStageFlags::TRANSFER,
    //             PipelineStageFlags::FRAGMENT_SHADER,
    //             DependencyFlags::empty(),
    //             &[],
    //             &[],
    //             &[*ImageMemoryBarrier::builder()
    //                 .src_access_mask(AccessFlags::MEMORY_READ)
    //                 .dst_access_mask(AccessFlags::MEMORY_WRITE)
    //                 .image(img.image)
    //                 .old_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
    //                 .new_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
    //                 .subresource_range(img_subresource_range)],
    //         );
    //     }
    //
    //     self.resource_loader.work_buffers.push(work_buffer);
    // }

    pub fn pipeline_render_create_info(&self) -> Option<PipelineRenderingCreateInfo> {
        match self.renderstate {
            RenderState::Dynamic {
                ref color_attachments,
                depth_attachments,
                stencil_attachments,
            } => Some(
                *PipelineRenderingCreateInfo::builder()
                    .color_attachment_formats(color_attachments)
                    .depth_attachment_format(depth_attachments[0])
                    .stencil_attachment_format(stencil_attachments[0]),
            ),
            RenderState::Renderpass(_) => None,
        }
    }

    pub fn logical(&self) -> &ash::Device {
        &self.device_state.device
    }

    pub fn logical_raw(&self) -> *const ash::Device {
        &self.device_state.device as *const _
    }

    pub fn physical(&self) -> ash::vk::PhysicalDevice {
        self.device_state.physical.device
    }

    pub fn debug_set_object_name<T: VkObjectType + ash::vk::Handle>(
        &self,
        vkobject: T,
        name: &str,
    ) {
        let _ = unsafe {
            self.dbg.set_debug_utils_object_name(
                self.logical().handle(),
                &ash::vk::DebugUtilsObjectNameInfoEXT::builder()
                    .object_handle(vkobject.as_raw())
                    .object_name(&std::ffi::CString::new(name).unwrap()),
            )
        };
    }

    pub fn debug_marker_begin(&self, cmd_buf: CommandBuffer, name: &str, color: [f32; 4]) {
        let _ = unsafe {
            self.dbg.cmd_begin_debug_utils_label(
                cmd_buf,
                &DebugUtilsLabelEXT::builder()
                    .label_name(&std::ffi::CString::new(name).unwrap())
                    .color(color),
            )
        };
    }

    pub fn debug_insert_label(&self, cmd_buf: CommandBuffer, name: &str, color: [f32; 4]) {
        let _ = unsafe {
            self.dbg.cmd_insert_debug_utils_label(
                cmd_buf,
                &DebugUtilsLabelEXT::builder()
                    .label_name(&std::ffi::CString::new(name).unwrap())
                    .color(color),
            )
        };
    }

    pub fn debug_marker_end(&self, cmd_buf: CommandBuffer) {
        let _ = unsafe { self.dbg.cmd_end_debug_utils_label(cmd_buf) };
    }

    pub fn debug_queue_begin_label(&self, name: &str, color: [f32; 4]) {
        let (_, queue, _, _, _) = self.device_state.queue_data(QueueType::Graphics);

        let _ = unsafe {
            self.dbg.queue_begin_debug_utils_label(
                queue,
                &DebugUtilsLabelEXT::builder()
                    .label_name(&std::ffi::CString::new(name).unwrap())
                    .color(color),
            )
        };
    }

    pub fn debug_queue_end_label(&self) {
        let (_, queue, _, _, _) = self.device_state.queue_data(QueueType::Graphics);
        let _ = unsafe { self.dbg.queue_end_debug_utils_label(queue) };
    }

    pub fn debug_queue_insert_label(&self, queue: Queue, name: &str, color: [f32; 4]) {
        // let (_, queue, _) = self.device_state.queue_data(QueueType::Graphics);
        let _ = unsafe {
            self.dbg.queue_insert_debug_utils_label(
                queue,
                &ash::vk::DebugUtilsLabelEXT::builder()
                    .label_name(&std::ffi::CString::new(name).unwrap())
                    .color(color),
            )
        };
    }
}

pub struct VulkanRenderer_DebugQueueScope {
    logical: *const VulkanRenderer,
}

impl VulkanRenderer_DebugQueueScope {
    pub fn begin(renderer: &VulkanRenderer, name: &str, color: [f32; 4]) -> Self {
        renderer.debug_queue_begin_label(name, color);
        Self {
            logical: renderer as *const _,
        }
    }
}

impl std::ops::Drop for VulkanRenderer_DebugQueueScope {
    fn drop(&mut self) {
        unsafe {
            (*self.logical).debug_queue_end_label();
        }
    }
}

pub struct VulkanRenderer_DebugMarkerScope {
    renderer: *const VulkanRenderer,
    cmd_buffer: CommandBuffer,
}

impl VulkanRenderer_DebugMarkerScope {
    pub fn begin(
        renderer: &VulkanRenderer,
        cmd_buffer: CommandBuffer,
        name: &str,
        color: [f32; 4],
    ) -> Self {
        renderer.debug_marker_begin(cmd_buffer, name, color);
        Self {
            renderer: renderer as *const _,
            cmd_buffer,
        }
    }
}

impl std::ops::Drop for VulkanRenderer_DebugMarkerScope {
    fn drop(&mut self) {
        unsafe {
            (*self.renderer).debug_marker_end(self.cmd_buffer);
        }
    }
}

impl std::ops::Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            self.device_state
                .device
                .unmap_memory(self.staging_sys.staging_memory);
            self.device_state
                .device
                .free_memory(self.staging_sys.staging_memory, None);
            self.device_state
                .device
                .destroy_buffer(self.staging_sys.staging_buffer, None);
        }
    }
}

pub struct DeviceProperties {
    pub base: PhysicalDeviceProperties2,
    pub vk11: PhysicalDeviceVulkan11Properties,
    pub vk12: PhysicalDeviceVulkan12Properties,
    pub vk13: PhysicalDeviceVulkan13Properties,
}

pub struct DeviceFeatures {
    pub base: PhysicalDeviceFeatures2,
    pub vk11: PhysicalDeviceVulkan11Features,
    pub vk12: PhysicalDeviceVulkan12Features,
    pub vk13: PhysicalDeviceVulkan13Features,
}

pub struct VulkanPhysicalDeviceState {
    pub device: PhysicalDevice,
    pub properties: DeviceProperties,
    pub memory_properties: PhysicalDeviceMemoryProperties,
    pub features: DeviceFeatures,
}

#[derive(Copy, Clone, Debug)]
#[repr(u8)]
pub enum QueueType {
    Graphics,
    Transfer,
}

pub struct QueueData {
    pub families: [u32; 2],
    pub queues: [ash::vk::Queue; 2],
    pub cmdpools: [ash::vk::CommandPool; 2],
    pub pool_locks: [SpinMutex; 2],
    pub queue_locks: [SpinMutex; 2],
}

pub struct VulkanDeviceState {
    pub pipeline_cache: PipelineCache,
    pub empty_descriptor_layout: DescriptorSetLayout,
    pub device: ash::Device,
    pub surface: VulkanSurfaceState,
    pub physical: VulkanPhysicalDeviceState,
    pub queues: QueueData,
}

impl std::ops::Drop for VulkanDeviceState {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.empty_descriptor_layout, None);
            self.device
                .destroy_pipeline_cache(self.pipeline_cache, None);
        }
    }
}

impl VulkanDeviceState {
    fn queue_data(
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
            self.queues.families[q as usize],
            self.queues.queues[q as usize],
            self.queues.cmdpools[q as usize],
            &self.queues.pool_locks[q as usize],
            &self.queues.queue_locks[q as usize],
        )
    }
}

pub struct VulkanSurfaceKHRState {
    pub ext: ash::extensions::khr::Surface,
    pub surface: SurfaceKHR,
}

pub struct VulkanSurfaceState {
    pub khr: VulkanSurfaceKHRState,
    pub caps: ash::vk::SurfaceCapabilitiesKHR,
    pub fmt: SurfaceFormatKHR,
    pub present_mode: PresentModeKHR,
    pub depth_fmt: ash::vk::Format,
}

pub struct VulkanSwapchainState {
    pub ext: ash::extensions::khr::Swapchain,
    pub swapchain: ash::vk::SwapchainKHR,
    pub images: Vec<Image>,
    pub image_views: Vec<ImageView>,
    pub framebuffers: Vec<Framebuffer>,
    pub depth_stencil: Vec<(Image, DeviceMemory)>,
    pub depth_stencil_views: Vec<ImageView>,
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
        previous_swapchain: Option<SwapchainKHR>,
        surface: &VulkanSurfaceState,
        device: &Device,
        physical: &VulkanPhysicalDeviceState,
        renderstate: RenderState,
        queue: u32,
    ) -> (
        SwapchainKHR,
        Vec<Image>,
        Vec<ImageView>,
        Vec<Framebuffer>,
        Vec<(Image, DeviceMemory)>,
        Vec<ImageView>,
    ) {
        let swapchain = unsafe {
            ext.create_swapchain(
                &SwapchainCreateInfoKHR::builder()
                    .surface(surface.khr.surface)
                    .min_image_count(surface.caps.max_image_count)
                    .image_format(surface.fmt.format)
                    .image_color_space(surface.fmt.color_space)
                    .image_extent(surface.caps.current_extent)
                    .image_array_layers(1)
                    .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
                    .image_sharing_mode(SharingMode::EXCLUSIVE)
                    .queue_family_indices(&[queue])
                    .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
                    .pre_transform(surface.caps.current_transform)
                    .old_swapchain(previous_swapchain.unwrap_or_else(|| SwapchainKHR::null()))
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

        let framebuffers = match renderstate {
            RenderState::Dynamic { .. } => vec![],
            RenderState::Renderpass(pass) => image_views
                .iter()
                .map(|&img_view| {
                    unsafe {
                        device.create_framebuffer(
                            &FramebufferCreateInfo::builder()
                                .render_pass(pass)
                                .attachments(&[img_view])
                                .width(surface.caps.current_extent.width)
                                .height(surface.caps.current_extent.height)
                                .layers(1),
                            None,
                        )
                    }
                    .expect("Failed to create framebuffer")
                })
                .collect::<Vec<_>>(),
        };

        //
        // create depth stencil images and image views
        let depth_stencil_images = image_views
            .iter()
            .map(|_| {
                let depth_stencil_image = unsafe {
                    device.create_image(
                        &ImageCreateInfo::builder()
                            .image_type(ImageType::TYPE_2D)
                            .format(surface.depth_fmt)
                            .extent(Extent3D {
                                width: surface.caps.current_extent.width,
                                height: surface.caps.current_extent.height,
                                depth: 1,
                            })
                            .mip_levels(1)
                            .array_layers(1)
                            .samples(SampleCountFlags::TYPE_1)
                            .tiling(ImageTiling::OPTIMAL)
                            .usage(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                            .sharing_mode(SharingMode::EXCLUSIVE)
                            .initial_layout(ImageLayout::UNDEFINED),
                        None,
                    )
                }
                .expect("Failed to create depth stencil image");

                let image_mem_req =
                    unsafe { device.get_image_memory_requirements(depth_stencil_image) };

                let depth_stencil_memory = unsafe {
                    device.allocate_memory(
                        &MemoryAllocateInfo::builder()
                            .allocation_size(image_mem_req.size)
                            .memory_type_index(choose_memory_heap(
                                &image_mem_req,
                                MemoryPropertyFlags::DEVICE_LOCAL,
                                &physical.memory_properties,
                            )),
                        None,
                    )
                }
                .expect("Depth stencil image memory allocation failed");

                unsafe { device.bind_image_memory(depth_stencil_image, depth_stencil_memory, 0) }
                    .expect("Failed to bind memory to image");

                (depth_stencil_image, depth_stencil_memory)
            })
            .collect::<Vec<_>>();

        let depth_stencil_imageviews = depth_stencil_images
            .iter()
            .map(|&(image, _)| unsafe {
                device
                    .create_image_view(
                        &ImageViewCreateInfo::builder()
                            .image(image)
                            .view_type(ImageViewType::TYPE_2D)
                            .format(surface.depth_fmt)
                            .components(*ComponentMapping::builder())
                            .subresource_range(
                                *ImageSubresourceRange::builder()
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
                    .expect("Failed to create depth stencil image view")
            })
            .collect::<Vec<_>>();

        (
            swapchain,
            images,
            image_views,
            framebuffers,
            depth_stencil_images,
            depth_stencil_imageviews,
        )
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
        renderstate: RenderState,
        previous_swapchain: Option<SwapchainKHR>,
    ) -> Option<VulkanSwapchainState> {
        let ext = ash::extensions::khr::Swapchain::new(instance, &ds.device);
        let (qid, _, cmd_pool, _, _) = ds.queue_data(QueueType::Graphics);

        let (swapchain, images, image_views, framebuffers, depth_stencil, depth_stencil_views) =
            Self::create_swapchain(
                &ext,
                previous_swapchain,
                &ds.surface,
                &ds.device,
                &ds.physical,
                renderstate,
                qid,
            );

        let (work_fences, sem_work_done, sem_img_available) =
            Self::create_sync_objects(&ds.device, ds.surface.caps.max_image_count);

        Some(VulkanSwapchainState {
            ext,
            swapchain,
            images,
            image_views,
            framebuffers,
            depth_stencil,
            depth_stencil_views,
            work_fences,
            sem_work_done,
            sem_img_available,
            cmd_buffers: unsafe {
                ds.device.allocate_command_buffers(
                    &CommandBufferAllocateInfo::builder()
                        .command_pool(cmd_pool)
                        .command_buffer_count(ds.surface.caps.max_image_count),
                )
            }
            .expect("Failed to allocate command buffers"),
            image_index: 0,
            max_frames: ds.surface.caps.max_image_count,
        })
    }

    pub fn handle_suboptimal(&mut self, ds: &VulkanDeviceState, renderpass: RenderState) {
        let (qid, queue, _, _, _) = ds.queue_data(QueueType::Graphics);
        unsafe {
            ds.device
                .queue_wait_idle(queue)
                .expect("Failed to wait queue idle");
            ds.device.device_wait_idle().expect("Failed to wait_idle()");
        }

        let recycled_swapchain = {
            let mut swapchain = SwapchainKHR::null();
            std::mem::swap(&mut self.swapchain, &mut swapchain);
            Some(swapchain)
        };

        let (
            swapchain,
            images,
            mut image_views,
            mut framebuffers,
            mut depth_stencil,
            mut depth_stencil_views,
        ) = Self::create_swapchain(
            &self.ext,
            recycled_swapchain,
            &ds.surface,
            &ds.device,
            &ds.physical,
            renderpass,
            qid,
        );

        self.swapchain = swapchain;
        self.images = images;

        std::mem::swap(&mut self.image_views, &mut image_views);
        image_views.into_iter().for_each(|img_view| unsafe {
            ds.device.destroy_image_view(img_view, None);
        });

        std::mem::swap(&mut self.framebuffers, &mut framebuffers);
        framebuffers.into_iter().for_each(|fb| unsafe {
            ds.device.destroy_framebuffer(fb, None);
        });

        std::mem::swap(&mut self.depth_stencil_views, &mut depth_stencil_views);
        depth_stencil_views.into_iter().for_each(|view| unsafe {
            ds.device.destroy_image_view(view, None);
        });

        std::mem::swap(&mut self.depth_stencil, &mut depth_stencil);
        depth_stencil
            .into_iter()
            .for_each(|(image, device_mem)| unsafe {
                ds.device.free_memory(device_mem, None);
                ds.device.destroy_image(image, None);
            });

        self.image_index = 0;

        let (mut fences, mut work_done, mut img_avail) =
            Self::create_sync_objects(&ds.device, self.max_frames);

        std::mem::swap(&mut self.work_fences, &mut fences);
        fences.into_iter().for_each(|fence| unsafe {
            ds.device.destroy_fence(fence, None);
        });

        std::mem::swap(&mut self.sem_work_done, &mut work_done);
        work_done.into_iter().for_each(|s| unsafe {
            ds.device.destroy_semaphore(s, None);
        });

        std::mem::swap(&mut self.sem_img_available, &mut img_avail);
        img_avail.into_iter().for_each(|s| unsafe {
            ds.device.destroy_semaphore(s, None);
        });
    }
}

impl VulkanRenderer {
    unsafe extern "system" fn debug_callback_stub(
        message_severity: ash::vk::DebugUtilsMessageSeverityFlagsEXT,
        _message_types: ash::vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const ash::vk::DebugUtilsMessengerCallbackDataEXT,
        _p_user_data: *mut std::os::raw::c_void,
    ) -> ash::vk::Bool32 {
        if message_severity.intersects(DebugUtilsMessageSeverityFlagsEXT::INFO) {
            log::info!(
                "[VK] {}",
                std::ffi::CStr::from_ptr((*p_callback_data).p_message)
                    .to_str()
                    .unwrap_or_else(|_| "unknown")
            );
        }

        if message_severity.intersects(DebugUtilsMessageSeverityFlagsEXT::WARNING) {
            log::warn!(
                "[VK] {}",
                std::ffi::CStr::from_ptr((*p_callback_data).p_message)
                    .to_str()
                    .unwrap_or_else(|_| "unknown")
            );
        }

        if message_severity.intersects(DebugUtilsMessageSeverityFlagsEXT::ERROR) {
            log::error!(
                "[VK] {}",
                std::ffi::CStr::from_ptr((*p_callback_data).p_message)
                    .to_str()
                    .unwrap_or_else(|_| "unknown")
            );
        }

        ash::vk::FALSE
    }

    #[cfg(target_os = "linux")]
    fn create_surface(
        entry: &Entry,
        instance: &Instance,
        wsi: WindowSystemIntegration,
    ) -> Result<VulkanSurfaceKHRState, ash::vk::Result> {
        let xlib_surface = ash::extensions::khr::XlibSurface::new(entry, instance);

        unsafe {
            xlib_surface.create_xlib_surface(
                &ash::vk::XlibSurfaceCreateInfoKHR::builder()
                    .dpy(wsi.native_disp as *mut ash::vk::Display)
                    .window(std::mem::transmute::<u64, ash::vk::Window>(wsi.native_win)),
                None,
            )
        }
        .map(|khr_surface| VulkanSurfaceKHRState {
            ext: ash::extensions::khr::Surface::new(entry, instance),
            surface: khr_surface,
        })
    }

    #[cfg(target_os = "windows")]
    fn create_surface(
        entry: &Entry,
        instance: &Instance,
        wsi: WindowSystemIntegration,
    ) -> VulkanSurfaceKHRState {
        let win32_surface = ash::extensions::khr::Win32Surface::new(entry, instance);

        let khr_surface = unsafe {
            win32_surface.create_win32_surface(
                &ash::vk::Win32SurfaceCreateInfoKHR::builder()
                    .hwnd(std::mem::transmute(wsi.hwnd))
                    .hinstance(std::mem::transmute(wsi.hinstance)),
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
        screen_size: (u32, u32),
    ) -> Result<VulkanDeviceState, ash::vk::Result> {
        let phys_devices = unsafe { instance.enumerate_physical_devices() }?;

        let mut phys_device: Option<(VulkanPhysicalDeviceState, u32, u32)> = None;
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

            let (phys_dev_props2, props_vk11, props_vk12, props_vk13) = unsafe {
                let mut props_vk11 = ash::vk::PhysicalDeviceVulkan11Properties::default();
                let mut props_vk12 = ash::vk::PhysicalDeviceVulkan12Properties::default();
                let mut props_vk13 = ash::vk::PhysicalDeviceVulkan13Properties::default();

                let mut phys_dev_props2 = ash::vk::PhysicalDeviceProperties2::builder()
                    .push_next(&mut props_vk11)
                    .push_next(&mut props_vk12)
                    .push_next(&mut props_vk13)
                    .build();

                instance.get_physical_device_properties2(pd, &mut phys_dev_props2);
                (phys_dev_props2, props_vk11, props_vk12, props_vk13)
            };

            let phys_dev_memory_props = unsafe {
                let mut props = ash::vk::PhysicalDeviceMemoryProperties2::default();
                instance.get_physical_device_memory_properties2(pd, &mut props);
                props
            };

            log::info!("{:?}", phys_dev_memory_props);

            log::info!(
                "{:?}\n{:?}\n{:?}\n{:?}",
                phys_dev_props2.properties,
                props_vk11,
                props_vk12,
                props_vk13
            );

            let (phys_dev_features2, f_vk11, f_vk12, f_vk13) = unsafe {
                let mut f_vk11 = ash::vk::PhysicalDeviceVulkan11Features::default();
                let mut f_vk12 = ash::vk::PhysicalDeviceVulkan12Features::default();
                let mut f_vk13 = ash::vk::PhysicalDeviceVulkan13Features::default();

                let mut pdf2 = ash::vk::PhysicalDeviceFeatures2::builder()
                    .push_next(&mut f_vk11)
                    .push_next(&mut f_vk12)
                    .push_next(&mut f_vk13)
                    .build();

                instance.get_physical_device_features2(pd, &mut pdf2);

                (pdf2, f_vk11, f_vk12, f_vk13)
            };

            log::info!(
                "{:?}\n{:?}\n{:?}\n{:?}",
                phys_dev_features2.features,
                f_vk11,
                f_vk12,
                f_vk13
            );

            let pd_features = unsafe { instance.get_physical_device_features(pd) };

            if pd_features.multi_draw_indirect == 0 || !pd_features.geometry_shader == 0 {
                log::info!(
                    "Rejecting device {} (no geometry shader and/or MultiDrawIndirect)",
                    pd_properties.device_id
                );
                continue;
            }

            if f_vk12.descriptor_binding_partially_bound == 0
                || f_vk12.descriptor_indexing == 0
                || f_vk12.draw_indirect_count == 0
            {
                log::info!(
                    "Rejecting device {} (no descriptor partially bound/descriptor indexing/draw indirect count)",
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
                    pd_properties.device_id,
                );
                continue;
            }

            //
            // query surface support
            let device_surface_support = unsafe {
                surface
                    .ext
                    .get_physical_device_surface_support(pd, queue_id, surface.surface)
            }?;

            if !device_surface_support {
                log::info!(
                    "Rejecting device {} (does not support surface)",
                    pd_properties.device_id
                );
                continue;
            }

            let surface_caps =
                Self::get_surface_capabilities(pd, surface.surface, &surface.ext, screen_size)?;

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
            }?;

            if maybe_surface_format.is_none() {
                log::info!("Rejecting device {} (does not support surface format B8G8R8A8_UNORM/R8G8B8A8_UNORM)", pd_properties.device_id);
                continue;
            }

            let image_format_props = unsafe {
                instance.get_physical_device_image_format_properties(
                    pd,
                    maybe_surface_format.unwrap().format,
                    ImageType::TYPE_2D,
                    ImageTiling::OPTIMAL,
                    ImageUsageFlags::SAMPLED,
                    ImageCreateFlags::empty(),
                )
            };

            if image_format_props.is_err() {
                log::info!(
                    "Rejecting device {} (image format properties error {:?})",
                    pd_properties.device_id,
                    image_format_props
                );
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

            //
            // query depth stencil support
            let depth_stencil_fmt = [Format::D32_SFLOAT_S8_UINT, Format::D24_UNORM_S8_UINT]
                .into_iter()
                .filter_map(|fmt| unsafe {
                    instance
                        .get_physical_device_image_format_properties(
                            pd,
                            fmt,
                            ImageType::TYPE_2D,
                            ImageTiling::OPTIMAL,
                            ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                            ImageCreateFlags::empty(),
                        )
                        .map(|_| fmt)
                        .ok()
                })
                .nth(0);

            if depth_stencil_fmt.is_none() {
                log::info!(
                    "Rejecting device {} (does not support depth stencil formats)",
                    pd_properties.device_id
                );
                continue;
            }

            surface_state = Some(VulkanSurfaceState {
                khr: surface,
                caps: surface_caps,
                fmt: maybe_surface_format.unwrap(),
                depth_fmt: depth_stencil_fmt.unwrap(),
                present_mode: maybe_present_mode.unwrap(),
            });

            let memory_properties = unsafe { instance.get_physical_device_memory_properties(pd) };

            phys_device = Some((
                VulkanPhysicalDeviceState {
                    device: pd,
                    properties: DeviceProperties {
                        base: ash::vk::PhysicalDeviceProperties2 {
                            p_next: std::ptr::null_mut(),
                            ..phys_dev_props2
                        },
                        vk11: ash::vk::PhysicalDeviceVulkan11Properties {
                            p_next: std::ptr::null_mut(),
                            ..props_vk11
                        },
                        vk12: ash::vk::PhysicalDeviceVulkan12Properties {
                            p_next: std::ptr::null_mut(),
                            ..props_vk12
                        },
                        vk13: ash::vk::PhysicalDeviceVulkan13Properties {
                            p_next: std::ptr::null_mut(),
                            ..props_vk13
                        },
                    },
                    features: DeviceFeatures {
                        base: ash::vk::PhysicalDeviceFeatures2 {
                            p_next: std::ptr::null_mut(),
                            ..phys_dev_features2
                        },
                        vk11: ash::vk::PhysicalDeviceVulkan11Features {
                            p_next: std::ptr::null_mut(),
                            ..f_vk11
                        },
                        vk12: ash::vk::PhysicalDeviceVulkan12Features {
                            p_next: std::ptr::null_mut(),
                            ..f_vk12
                        },
                        vk13: ash::vk::PhysicalDeviceVulkan13Features {
                            p_next: std::ptr::null_mut(),
                            ..f_vk13
                        },
                    },
                    memory_properties,
                },
                queue_id,
                maybe_transfer_queue.unwrap() as u32,
            ));

            break;
        }

        if surface_state.is_none() || phys_device.is_none() {
            return Err(ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }

        let surface_state = surface_state.unwrap();
        let (phys_device, graphics_queue, transfer_queue) = phys_device.unwrap();

        //
        // create logical device
        let device = unsafe {
            let mut f_vk11 = phys_device.features.vk11;
            let mut f_vk12 = phys_device.features.vk12;
            let mut f_vk13 = phys_device.features.vk13;

            instance.create_device(
                phys_device.device,
                &DeviceCreateInfo::builder()
                    .push_next(&mut f_vk11)
                    .push_next(&mut f_vk12)
                    .push_next(&mut f_vk13)
                    .enabled_extension_names(&[
                        ash::extensions::khr::Swapchain::name().as_ptr(),
                        ash::extensions::khr::DynamicRendering::name().as_ptr(),
                        ash::extensions::ext::DescriptorBuffer::name().as_ptr(),
                    ])
                    .queue_create_infos(&[
                        *DeviceQueueCreateInfo::builder()
                            .queue_family_index(graphics_queue)
                            .queue_priorities(&[1f32]),
                        *DeviceQueueCreateInfo::builder()
                            .queue_family_index(transfer_queue)
                            .queue_priorities(&[1f32]),
                    ]),
                None,
            )
        }?;

        let queues: SmallVec<[Queue; 2]> = [graphics_queue, transfer_queue]
            .iter()
            .map(|qid| unsafe { device.get_device_queue(*qid, 0) })
            .collect();

        let cmd_pools: SmallVec<[CommandPool; 2]> = [graphics_queue, transfer_queue]
            .iter()
            .filter_map(|qid| {
                unsafe {
                    device.create_command_pool(
                        &CommandPoolCreateInfo::builder()
                            .queue_family_index(*qid)
                            .flags(
                                CommandPoolCreateFlags::TRANSIENT
                                    | CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                            ),
                        None,
                    )
                }
                .ok()
            })
            .collect();

        if cmd_pools.len() != 2 {
            return Err(ash::vk::Result::ERROR_UNKNOWN);
        }

        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &DescriptorPoolCreateInfo::builder()
                    .max_sets(4096)
                    .pool_sizes(&[*DescriptorPoolSize::builder()
                        .ty(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .descriptor_count(1024)]),
                None,
            )
        }?;

        let empty_descriptor_layout = unsafe {
            device.create_descriptor_set_layout(&DescriptorSetLayoutCreateInfo::builder(), None)
        }?;

        Ok(VulkanDeviceState {
            pipeline_cache: unsafe {
                device.create_pipeline_cache(&ash::vk::PipelineCacheCreateInfo::builder(), None)
            }?,
            device,
            physical: phys_device,
            surface: surface_state,
            empty_descriptor_layout,
            queues: QueueData {
                families: [graphics_queue, transfer_queue],
                queues: [queues[0], queues[1]],
                cmdpools: [cmd_pools[0], cmd_pools[1]],
                pool_locks: [SpinMutex::new(), SpinMutex::new()],
                queue_locks: [SpinMutex::new(), SpinMutex::new()],
            },
        })
    }

    pub fn new(wsi: WindowSystemIntegration, window_size: (u32, u32)) -> Option<VulkanRenderer> {
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
            .pfn_user_callback(Some(VulkanRenderer::debug_callback_stub));

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
            ash::extensions::khr::Surface::name().as_ptr(),
            #[cfg(target_os = "linux")]
            ash::extensions::khr::XlibSurface::name().as_ptr(),
            #[cfg(target_os = "windows")]
            ash::extensions::khr::Win32Surface::name().as_ptr(),
            ash::extensions::ext::DebugUtils::name().as_ptr(),
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

        let surface_state = Self::create_surface(&entry, &instance, wsi).ok()?;

        let device_state =
            Self::pick_device(&instance, surface_state, window_size).expect("Faile to pick device");

        let renderstate =
            Self::create_render_state(&device_state).expect("Failed to create render_state");

        let swapchain_state =
            VulkanSwapchainState::new(&instance, &device_state, renderstate, None)?;

        let staging_sys = StagingSystem::create(
            &device_state.device,
            &device_state.physical.memory_properties,
        )
        .ok()?;

        Some(VulkanRenderer {
            staging_sys,
            dbg,
            msgr,
            entry,
            instance,
            device_state,
            swapchain: swapchain_state,
            renderstate,
        })
    }

    fn create_render_state(ds: &VulkanDeviceState) -> VkResult<RenderState> {
        if ds.physical.features.vk13.dynamic_rendering != 0 {
            Ok(RenderState::Dynamic {
                color_attachments: [ds.surface.fmt.format],
                depth_attachments: [ds.surface.depth_fmt],
                stencil_attachments: [ds.surface.depth_fmt],
            })
        } else {
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
            .map(|vk_pass| RenderState::Renderpass(vk_pass))
        }
    }

    pub fn begin_rendering(&mut self, fb_size: Extent2D) -> FrameRenderContext {
        //
        // wait for previous submittted work
        unsafe {
            self.device_state
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
        let swapchain_available_img_index = 'acquire_next_image: loop {
            match unsafe {
                self.swapchain.ext.acquire_next_image(
                    self.swapchain.swapchain,
                    u64::MAX,
                    self.swapchain.sem_img_available[self.swapchain.image_index as usize],
                    Fence::null(),
                )
            } {
                Err(e) => {
                    if e == ash::vk::Result::ERROR_OUT_OF_DATE_KHR {
                        log::info!("Swapchain out of date, recreating ...");
                        self.handle_surface_size_changed(fb_size);
                        self.swapchain
                            .handle_suboptimal(&self.device_state, self.renderstate);
                    } else {
                        log::error!("Present error: {:?}", e);
                        todo!("Handle this ...");
                    }
                }
                Ok((swapchain_available_img_index, suboptimal)) => {
                    if suboptimal {
                        log::info!("Swapchain suboptimal, recreating ...");
                        self.handle_surface_size_changed(fb_size);
                        self.swapchain
                            .handle_suboptimal(&self.device_state, self.renderstate);
                    } else {
                        break 'acquire_next_image swapchain_available_img_index;
                    }
                }
            }
        };

        unsafe {
            self.device_state
                .device
                .reset_fences(&[self.swapchain.work_fences[self.swapchain.image_index as usize]])
                .expect("Failed to reset fence ...");
        }

        //
        // begin command buffer + renderpass

        unsafe {
            self.device_state
                .device
                .reset_command_buffer(
                    self.swapchain.cmd_buffers[self.swapchain.image_index as usize],
                    CommandBufferResetFlags::empty(),
                )
                .expect("Failed to reset command buffer");

            self.device_state
                .device
                .begin_command_buffer(
                    self.swapchain.cmd_buffers[self.swapchain.image_index as usize],
                    &CommandBufferBeginInfo::builder()
                        .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("Failed to begin command buffer ...");
        }

        let color_clear = ClearValue {
            color: ClearColorValue {
                float32: [0f32, 0f32, 0f32, 1f32],
            },
        };

        let depth_stencil_clear = ClearValue {
            depth_stencil: ClearDepthStencilValue {
                depth: 1.0f32,
                stencil: 0,
            },
        };

        let (qid, queue, cmd_pool, _, _) = self.device_state.queue_data(QueueType::Graphics);
        match self.renderstate {
            RenderState::Dynamic { .. } => unsafe {
                //
                // transition attachments from undefined layout to optimal layout

                self.device_state.device.cmd_pipeline_barrier2(
                    self.swapchain.cmd_buffers[self.swapchain.image_index as usize],
                    &DependencyInfo::builder()
                        .dependency_flags(DependencyFlags::BY_REGION)
                        .image_memory_barriers(&[
                            *ImageMemoryBarrier2::builder()
                                .src_stage_mask(PipelineStageFlags2::TOP_OF_PIPE)
                                .src_access_mask(AccessFlags2::NONE)
                                .dst_stage_mask(PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                                .dst_access_mask(AccessFlags2::COLOR_ATTACHMENT_WRITE)
                                .old_layout(ImageLayout::UNDEFINED)
                                .new_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                                .src_queue_family_index(qid)
                                .dst_queue_family_index(qid)
                                .image(
                                    self.swapchain.images[swapchain_available_img_index as usize],
                                )
                                .subresource_range(
                                    *ImageSubresourceRange::builder()
                                        .aspect_mask(ImageAspectFlags::COLOR)
                                        .base_mip_level(0)
                                        .level_count(1)
                                        .base_array_layer(0)
                                        .layer_count(1),
                                ),
                            *ImageMemoryBarrier2::builder()
                                .src_stage_mask(PipelineStageFlags2::TOP_OF_PIPE)
                                .src_access_mask(AccessFlags2::NONE)
                                .dst_stage_mask(PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                                .dst_access_mask(AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                                .old_layout(ImageLayout::UNDEFINED)
                                .new_layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                                .src_queue_family_index(qid)
                                .dst_queue_family_index(qid)
                                .image(
                                    self.swapchain.depth_stencil
                                        [self.swapchain.image_index as usize]
                                        .0,
                                )
                                .subresource_range(
                                    *ImageSubresourceRange::builder()
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
                self.device_state.device.cmd_begin_rendering(
                    self.swapchain.cmd_buffers[self.swapchain.image_index as usize],
                    &RenderingInfo::builder()
                        .render_area(Rect2D {
                            offset: Offset2D::default(),
                            extent: fb_size,
                        })
                        .layer_count(1)
                        .color_attachments(&[*RenderingAttachmentInfo::builder()
                            .image_view(
                                self.swapchain.image_views[swapchain_available_img_index as usize],
                            )
                            .image_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(AttachmentLoadOp::CLEAR)
                            .store_op(AttachmentStoreOp::STORE)
                            .clear_value(color_clear)])
                        .depth_attachment(
                            &RenderingAttachmentInfo::builder()
                                .image_view(
                                    self.swapchain.depth_stencil_views
                                        [self.swapchain.image_index as usize],
                                )
                                .image_layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                                .load_op(AttachmentLoadOp::CLEAR)
                                .store_op(AttachmentStoreOp::STORE)
                                .clear_value(depth_stencil_clear),
                        )
                        .stencil_attachment(
                            &RenderingAttachmentInfo::builder()
                                .image_view(
                                    self.swapchain.depth_stencil_views
                                        [self.swapchain.image_index as usize],
                                )
                                .image_layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                                .load_op(AttachmentLoadOp::CLEAR)
                                .store_op(AttachmentStoreOp::STORE)
                                .clear_value(depth_stencil_clear),
                        ),
                );
            },
            RenderState::Renderpass(pass) => unsafe {
                self.device_state.device.cmd_begin_render_pass(
                    self.swapchain.cmd_buffers[self.swapchain.image_index as usize],
                    &RenderPassBeginInfo::builder()
                        .render_pass(pass)
                        .framebuffer(
                            self.swapchain.framebuffers[swapchain_available_img_index as usize],
                        )
                        .render_area(Rect2D {
                            offset: Offset2D::default(),
                            extent: fb_size,
                        })
                        .clear_values(&[color_clear, depth_stencil_clear]),
                    SubpassContents::INLINE,
                );
            },
        }

        FrameRenderContext {
            cmd_buff: self.swapchain.cmd_buffers[self.swapchain.image_index as usize],
            fb_size,
            current_frame_id: self.swapchain.image_index,
            swapchain_image_index: swapchain_available_img_index,
        }
    }

    pub fn end_rendering(&mut self, frame_ctx: FrameRenderContext) {
        let (qid, queue, cmd_pool, _, _) = self.device_state.queue_data(QueueType::Graphics);
        //
        // end command buffer + renderpass
        match self.renderstate {
            RenderState::Renderpass(_) => unsafe {
                self.device_state
                    .device
                    .cmd_end_render_pass(frame_ctx.cmd_buff);
            },
            RenderState::Dynamic { .. } => unsafe {
                self.device_state
                    .device
                    .cmd_end_rendering(frame_ctx.cmd_buff);
                //
                // transition image from attachment optimal to SRC_PRESENT
                self.device_state.device.cmd_pipeline_barrier2(
                    frame_ctx.cmd_buff,
                    &DependencyInfo::builder()
                        .dependency_flags(DependencyFlags::BY_REGION)
                        .image_memory_barriers(&[*ImageMemoryBarrier2::builder()
                            .src_stage_mask(PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                            .src_access_mask(AccessFlags2::COLOR_ATTACHMENT_WRITE)
                            .dst_stage_mask(PipelineStageFlags2::BOTTOM_OF_PIPE)
                            .dst_access_mask(AccessFlags2::MEMORY_READ)
                            .old_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .new_layout(ImageLayout::PRESENT_SRC_KHR)
                            .src_queue_family_index(qid)
                            .dst_queue_family_index(qid)
                            .image(self.swapchain.images[frame_ctx.swapchain_image_index as usize])
                            .subresource_range(
                                *ImageSubresourceRange::builder()
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
                .device
                .end_command_buffer(self.swapchain.cmd_buffers[self.swapchain.image_index as usize])
                .expect("Failed to end command buffer");
        }

        //
        // submit
        unsafe {
            self.device_state
                .device
                .queue_submit(
                    queue,
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
                queue,
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
                        self.handle_surface_size_changed(frame_ctx.fb_size);
                        self.swapchain
                            .handle_suboptimal(&self.device_state, self.renderstate);
                    } else {
                        log::error!("Present error: {:?}", e);
                        todo!("Handle this ...");
                    }
                }
                Ok(suboptimal) => {
                    if suboptimal {
                        log::info!("Swapchain suboptimal, recreating ...");
                        self.handle_surface_size_changed(frame_ctx.fb_size);
                        self.swapchain
                            .handle_suboptimal(&self.device_state, self.renderstate);
                    } else {
                        self.swapchain.image_index =
                            (self.swapchain.image_index + 1) % self.swapchain.max_frames;
                    }
                }
            };
        }
    }

    pub fn handle_surface_size_changed(&mut self, surface_size: Extent2D) {
        self.device_state.surface.caps = Self::get_surface_capabilities(
            self.device_state.physical.device,
            self.device_state.surface.khr.surface,
            &self.device_state.surface.khr.ext,
            (surface_size.width, surface_size.height),
        )
        .expect("Failed to query surface capabilities");
        log::info!(
            "Surface extent {:?}",
            self.device_state.surface.caps.current_extent
        );
    }

    fn get_surface_capabilities(
        phys_device: PhysicalDevice,
        surface: SurfaceKHR,
        surface_ext: &ash::extensions::khr::Surface,
        screen_size: (u32, u32),
    ) -> VkResult<ash::vk::SurfaceCapabilitiesKHR> {
        unsafe { surface_ext.get_physical_device_surface_capabilities(phys_device, surface) }.map(
            |surface_caps| {
                let current_extent = if surface_caps.current_extent.width == 0xffffffff
                    || surface_caps.current_extent.height == 0xffffffff
                {
                    //
                    // width and height determined by the swapchain's extent
                    ash::vk::Extent2D {
                        width: screen_size.0,
                        height: screen_size.1,
                    }
                } else {
                    surface_caps.current_extent
                };

                let max_image_count = if surface_caps.max_image_count == 0 {
                    //
                    // no limit on the number of images
                    (surface_caps.min_image_count as f32 * 1.5f32).ceil() as u32
                } else {
                    (surface_caps.min_image_count + 2).min(surface_caps.max_image_count)
                };

                log::info!(
                    "Surface caps: extent: {:?}, image count {}",
                    current_extent,
                    max_image_count
                );

                ash::vk::SurfaceCapabilitiesKHR {
                    max_image_count,
                    current_extent,
                    ..surface_caps
                }
            },
        )
    }

    // pub fn create_staging_buffer(
    //     &mut self,
    //     bytes_size: usize,
    // ) -> std::result::Result<BufferWithBoundDeviceMemory, GraphicsError> {
    //     let bytes_size = round_up(bytes_size, self.limits().non_coherent_atom_size as usize) as u64;
    //
    //     let buffer = unsafe {
    //         self.logical().create_buffer(
    //             &BufferCreateInfo::builder()
    //                 .usage(BufferUsageFlags::TRANSFER_SRC)
    //                 .size(bytes_size as u64)
    //                 .sharing_mode(SharingMode::EXCLUSIVE),
    //             None,
    //         )
    //     }?;
    //
    //     let device: *const Device = self.logical() as *const _;
    //     scopeguard::defer_on_unwind! {
    //         unsafe {
    //             (*device).destroy_buffer(buffer, None);
    //         }
    //     };
    //
    //     let memory_requirements = unsafe {
    //         let mut memory_requirements = std::mem::MaybeUninit::<MemoryRequirements2>::uninit();
    //
    //         self.logical().get_buffer_memory_requirements2(
    //             &BufferMemoryRequirementsInfo2::builder().buffer(buffer),
    //             &mut *memory_requirements.as_mut_ptr(),
    //         );
    //
    //         memory_requirements.assume_init()
    //     };
    //
    //     let buffer_memory = unsafe {
    //         self.logical().allocate_memory(
    //             &MemoryAllocateInfo::builder()
    //                 .allocation_size(memory_requirements.memory_requirements.size)
    //                 .memory_type_index(choose_memory_heap(
    //                     &memory_requirements.memory_requirements,
    //                     MemoryPropertyFlags::DEVICE_LOCAL | MemoryPropertyFlags::HOST_VISIBLE,
    //                     self.memory_properties(),
    //                 )),
    //             None,
    //         )
    //     }?;
    //
    //     scopeguard::defer_on_unwind! {
    //         unsafe {
    //             (*device).free_memory(buffer_memory, None);
    //         }
    //     };
    //
    //     unsafe { self.logical().bind_buffer_memory(buffer, buffer_memory, 0) }?;
    //
    //     Ok(BufferWithBoundDeviceMemory(buffer, buffer_memory))
    // }
}

pub struct FrameRenderContext {
    pub cmd_buff: CommandBuffer,
    pub fb_size: Extent2D,
    pub current_frame_id: u32,
    pub swapchain_image_index: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, enum_iterator::Sequence)]
#[repr(u8)]
pub enum BindlessResourceType {
    UniformBuffer,
    Ssbo,
    CombinedImageSampler,
}

impl BindlessResourceType {
    fn as_vk_type(&self) -> ash::vk::DescriptorType {
        match self {
            BindlessResourceType::Ssbo => DescriptorType::STORAGE_BUFFER,
            BindlessResourceType::CombinedImageSampler => DescriptorType::COMBINED_IMAGE_SAMPLER,
            BindlessResourceType::UniformBuffer => DescriptorType::UNIFORM_BUFFER,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BindlessResourceHandle(u32);

impl BindlessResourceHandle {
    pub fn get_type(&self) -> BindlessResourceType {
        let bits = self.0 & 0b11;
        match bits {
            0 => BindlessResourceType::Ssbo,
            1 => BindlessResourceType::CombinedImageSampler,
            2 => BindlessResourceType::UniformBuffer,
            _ => todo!("Handle this"),
        }
    }

    pub fn get_id(&self) -> u32 {
        self.0 >> 2
    }

    pub fn offset(&self, frame: u32) -> BindlessResourceHandle {
        Self::new(self.get_type(), self.get_id() + frame)
    }

    pub fn new(ty: BindlessResourceType, id: u32) -> Self {
        match ty {
            BindlessResourceType::Ssbo => Self(id << 2),
            BindlessResourceType::CombinedImageSampler => Self(1 | (id << 2)),
            BindlessResourceType::UniformBuffer => Self(2 | (id << 2)),
        }
    }
}

/// Format is
/// [0..4] frame id [5 .. 15] buffer id
pub struct GlobalPushConstant(u32);

impl GlobalPushConstant {
    pub fn from_resource(resource: BindlessResourceHandle, frame_id: u32) -> Self {
        assert!(frame_id < (1 << 4));
        assert!(resource.get_id() < 0x7ff);
        GlobalPushConstant(resource.get_id() << 4 | frame_id)
    }

    pub fn to_gpu(&self) -> [u8; 4] {
        self.0.to_le_bytes()
    }
}

impl std::convert::AsRef<[u8]> for GlobalPushConstant {
    fn as_ref(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(&self.0 as *const _ as *const u8, std::mem::size_of::<u32>())
        }
    }
}

#[derive(Copy, Clone)]
struct WriteBuffer {
    buffer: Buffer,
    slabs: usize,
    slab_size: usize,
    idx: u32,
}

pub struct BindlessResourceSystem {
    dpool: DescriptorPool,
    set_layouts: Vec<DescriptorSetLayout>,
    descriptor_sets: Vec<DescriptorSet>,
    ssbos: Vec<BindlessResourceHandle>,
    samplers: Vec<BindlessResourceHandle>,
    ubos: Vec<BindlessResourceHandle>,
    bindless_pipeline_layout: PipelineLayout,
    ubo_idx: u32,
    ssbo_idx: u32,
    ubo_pending_writes: Vec<WriteBuffer>,
    ssbo_pending_writes: Vec<WriteBuffer>,
}

impl BindlessResourceSystem {
    pub fn descriptor_sets(&self) -> &[DescriptorSet] {
        &self.descriptor_sets
    }
    pub fn pipeline_layout(&self) -> PipelineLayout {
        self.bindless_pipeline_layout
    }

    pub fn make_push_constant(frame: u32, resource: BindlessResourceHandle) -> u32 {
        let res_id = resource.get_id();
        frame << 16 | res_id & 0xFFFF
    }

    pub fn new(vks: &VulkanRenderer) -> Self {
        let dpool_sizes = [
            *DescriptorPoolSize::builder()
                .ty(DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1024),
            *DescriptorPoolSize::builder()
                .ty(DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1024),
            *DescriptorPoolSize::builder()
                .ty(DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1024),
        ];

        let dpool = unsafe {
            vks.device_state.device.create_descriptor_pool(
                &*DescriptorPoolCreateInfo::builder()
                    .flags(ash::vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                    .pool_sizes(&dpool_sizes)
                    .max_sets(8),
                None,
            )
        }
        .expect("Failed to create bindless dpool");

        use enum_iterator::all;

        let set_layouts = all::<BindlessResourceType>()
            .map(|res_type| {
                unsafe {
                    let mut flag_info =
                        *ash::vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                            .binding_flags(&[ash::vk::DescriptorBindingFlags::PARTIALLY_BOUND]);

                    let dcount = match res_type {
                        BindlessResourceType::UniformBuffer => 8,
                        _ => 1024,
                    };

                    vks.device_state.device.create_descriptor_set_layout(
                        &DescriptorSetLayoutCreateInfo::builder()
                            .push_next(&mut flag_info)
                            .flags(ash::vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                            .bindings(&[*ash::vk::DescriptorSetLayoutBinding::builder()
                                .descriptor_count(dcount)
                                .binding(0)
                                .descriptor_type(res_type.as_vk_type())
                                .stage_flags(ash::vk::ShaderStageFlags::ALL)]),
                        None,
                    )
                }
                .expect("Failed to create layout")
            })
            .collect::<Vec<_>>();

        let descriptor_sets = unsafe {
            vks.device_state.device.allocate_descriptor_sets(
                &DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(dpool)
                    .set_layouts(&set_layouts),
            )
        }
        .expect("Failed to allocate bindless descriptor sets");

        let bindless_pipeline_layout = unsafe {
            vks.device_state.device.create_pipeline_layout(
                &PipelineLayoutCreateInfo::builder()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&[*PushConstantRange::builder()
                        .stage_flags(ShaderStageFlags::ALL)
                        .offset(0)
                        .size(size_of::<u32>() as _)]),
                None,
            )
        }
        .expect("Failed to create bindless pipeline layout");

        Self {
            dpool,
            set_layouts,
            descriptor_sets,
            ssbos: vec![],
            samplers: vec![],
            ubos: vec![],
            bindless_pipeline_layout,
            ssbo_pending_writes: vec![],
            ssbo_idx: 0u32,
            ubo_idx: 0u32,
            ubo_pending_writes: vec![],
        }
    }

    pub fn register_ssbo(
        &mut self,
        vks: &VulkanDeviceState,
        ssbo: &UniqueBuffer,
    ) -> BindlessResourceHandle {
        let idx = self.ssbos.len() as u32;
        let handle = BindlessResourceHandle::new(BindlessResourceType::Ssbo, idx);
        self.ssbos.push(handle);

        unsafe {
            let buf_info = *DescriptorBufferInfo::builder()
                .buffer(ssbo.handle)
                .range(WHOLE_SIZE)
                .offset(0);
            let write = *WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[BindlessResourceType::Ssbo as usize])
                .dst_binding(0)
                .dst_array_element(idx)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info));

            vks.device
                .update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }

        handle
    }

    pub fn register_image(
        &mut self,
        vks: &VulkanDeviceState,
        imgview: &UniqueImageView,
        sampler: &UniqueSampler,
    ) -> BindlessResourceHandle {
        let idx = self.samplers.len() as u32;
        let handle = BindlessResourceHandle::new(BindlessResourceType::CombinedImageSampler, idx);
        self.samplers.push(handle);

        unsafe {
            let img_info = *DescriptorImageInfo::builder()
                .image_view(imgview.view)
                .sampler(sampler.handle)
                .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            let write = *WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[BindlessResourceType::CombinedImageSampler as usize])
                .dst_binding(0)
                .dst_array_element(idx)
                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&img_info));

            vks.device
                .update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }

        handle
    }

    fn register_ubo_impl(&mut self, vks: &VulkanRenderer, ubo: Buffer) -> BindlessResourceHandle {
        let idx = self.ubos.len() as u32;
        let handle = BindlessResourceHandle::new(BindlessResourceType::UniformBuffer, idx);
        self.ubos.push(handle);

        let buf_info = *DescriptorBufferInfo::builder()
            .buffer(ubo)
            .range(WHOLE_SIZE)
            .offset(0);

        unsafe {
            let write = *WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[BindlessResourceType::UniformBuffer as usize])
                .dst_binding(0)
                .dst_array_element(idx)
                .descriptor_type(DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info));

            vks.logical()
                .update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }

        handle
    }

    pub fn register_uniform_buffer(
        &mut self,
        vks: &VulkanRenderer,
        ubo: &UniqueBuffer,
    ) -> BindlessResourceHandle {
        self.register_ubo_impl(vks, ubo.handle)
    }

    pub fn register_uniform_buffer_vkbuffer(
        &mut self,
        ubo: &VulkanBuffer,
    ) -> BindlessResourceHandle {
        let idx = self.ubo_idx;
        self.ubo_idx += ubo.slabs as u32;

        let handle = BindlessResourceHandle::new(BindlessResourceType::UniformBuffer, idx);
        self.ubo_pending_writes.push(WriteBuffer {
            buffer: ubo.buffer,
            slabs: ubo.slabs,
            slab_size: ubo.aligned_slab_size,
            idx,
        });

        handle
    }

    pub fn register_storage_buffer(&mut self, ssbo: &VulkanBuffer) -> BindlessResourceHandle {
        let idx = self.ssbo_idx;
        self.ssbo_idx += ssbo.slabs as u32;

        let handle = BindlessResourceHandle::new(BindlessResourceType::Ssbo, idx);
        self.ssbo_pending_writes.push(WriteBuffer {
            buffer: ssbo.buffer,
            slabs: ssbo.slabs,
            slab_size: ssbo.aligned_slab_size,
            idx,
        });

        handle
    }

    pub fn flush_pending_updates(&mut self, vks: &VulkanRenderer) {
        let pending_writes = self
            .ubo_pending_writes
            .iter()
            .chain(self.ssbo_pending_writes.iter())
            .map(|pending_write| pending_write.slabs)
            .sum();

        if pending_writes == 0 {
            return;
        }

        let mut buffer_infos = SmallVec::<[DescriptorBufferInfo; 8]>::with_capacity(pending_writes);
        let mut descriptor_writes = SmallVec::<[WriteDescriptorSet; 2]>::new();

        if !self.ubo_pending_writes.is_empty() {
            let dst_array_element = self.ubo_pending_writes[0].idx;

            for pending_write in self.ubo_pending_writes.drain(..) {
                buffer_infos.extend((0..pending_write.slabs).map(|slab| {
                    *DescriptorBufferInfo::builder()
                        .buffer(pending_write.buffer)
                        .offset((slab * pending_write.slab_size) as DeviceSize)
                        .range(pending_write.slab_size as DeviceSize)
                }));
            }

            descriptor_writes.push(
                *WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[BindlessResourceType::UniformBuffer as usize])
                    .dst_binding(0)
                    .dst_array_element(dst_array_element)
                    .descriptor_type(DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_infos[0..]),
            );
        }

        if !self.ssbo_pending_writes.is_empty() {
            let dst_array_element = self.ssbo_pending_writes[0].idx;
            let ssbo_offset = buffer_infos.len();

            for pending_write in self.ssbo_pending_writes.drain(..) {
                buffer_infos.extend((0..pending_write.slabs).map(|slab| {
                    *DescriptorBufferInfo::builder()
                        .buffer(pending_write.buffer)
                        .offset((slab * pending_write.slab_size) as DeviceSize)
                        .range(pending_write.slab_size as DeviceSize)
                }));
            }

            descriptor_writes.push(
                *WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[BindlessResourceType::Ssbo as usize])
                    .dst_binding(0)
                    .dst_array_element(dst_array_element)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_infos[ssbo_offset..]),
            );
        }

        dbg!(&buffer_infos);
        dbg!(&descriptor_writes);

        unsafe {
            vks.logical()
                .update_descriptor_sets(&descriptor_writes, &[]);
        }
    }

    pub fn bind_descriptors(&self, cmd_buff: CommandBuffer, vks: &VulkanRenderer) {
        unsafe {
            vks.logical().cmd_bind_descriptor_sets(
                cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.bindless_pipeline_layout,
                0,
                &self.descriptor_sets,
                &[],
            )
        }
    }
}

pub struct InputAssemblyState {
    pub stride: u32,
    pub input_rate: VertexInputRate,
    pub vertex_descriptions: Vec<VertexInputAttributeDescription>,
}

#[derive(Default)]
pub struct GraphicsPipelineSetupHelper<'a> {
    shader_stages: Vec<ShaderSource<'a>>,
    input_assembly_state: Option<InputAssemblyState>,
    rasterization_state: Option<PipelineRasterizationStateCreateInfo>,
    multisample_state: Option<PipelineMultisampleStateCreateInfo>,
    depth_stencil_state: Option<PipelineDepthStencilStateCreateInfo>,
    color_blend_state: Option<PipelineColorBlendStateCreateInfo>,
    dynamic_state: Vec<DynamicState>,
}

pub struct GraphicsPipelineCreateOptions {
    pub layout: Option<PipelineLayout>,
}

impl<'a> GraphicsPipelineSetupHelper<'a> {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    pub fn set_input_assembly_state(mut self, state: InputAssemblyState) -> Self {
        self.input_assembly_state = Some(state);
        self
    }

    pub fn add_shader_stage(mut self, stage: ShaderSource<'a>) -> Self {
        self.shader_stages.push(stage);
        self
    }

    pub fn set_raster_state(mut self, raster: PipelineRasterizationStateCreateInfo) -> Self {
        self.rasterization_state = Some(raster);
        self
    }

    pub fn set_multisample_state(mut self, ms: PipelineMultisampleStateCreateInfo) -> Self {
        self.multisample_state = Some(ms);
        self
    }

    pub fn set_depth_stencil_state(mut self, ds: PipelineDepthStencilStateCreateInfo) -> Self {
        self.depth_stencil_state = Some(ds);
        self
    }

    pub fn set_colorblend_state(mut self, cb: PipelineColorBlendStateCreateInfo) -> Self {
        self.color_blend_state = Some(cb);
        self
    }

    pub fn set_dynamic_state(mut self, ds: &[DynamicState]) -> Self {
        self.dynamic_state.extend(ds.iter());
        self
    }

    pub fn create(
        self,
        renderer: &VulkanRenderer,
        options: GraphicsPipelineCreateOptions,
    ) -> std::result::Result<UniquePipeline, GraphicsError> {
        assert!(!self.shader_stages.is_empty());

        let mut reflected_vertex_inputs_attribute_descriptions: Option<(
            u32,
            Vec<VertexInputAttributeDescription>,
        )> = None;

        let shader_entry_point = std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap();

        let (shader_modules_data, pipeline_shader_stage_create_info) = {
            use crate::shader::*;

            let mut module_data: Vec<(UniqueShaderModule, ShaderReflection)> = vec![];
            let mut shader_stage_create_info: Vec<PipelineShaderStageCreateInfo> = vec![];

            for shader_src in self.shader_stages.into_iter() {
                let (shader_module, shader_stage, shader_reflection) = compile_and_reflect_shader(
                    renderer.logical(),
                    &crate::shader::ShaderCompileInfo {
                        src: &shader_src,
                        entry_point: Some("main"),
                        compile_defs: &[],
                        optimize: false,
                        debug_info: true,
                    },
                )?;

                dbg!(&shader_reflection);

                shader_stage_create_info.push(
                    *PipelineShaderStageCreateInfo::builder()
                        .stage(shader_stage)
                        .module(*shader_module)
                        .name(&shader_entry_point),
                );

                if shader_stage == ShaderStageFlags::VERTEX && !shader_reflection.inputs.is_empty()
                {
                    assert!(reflected_vertex_inputs_attribute_descriptions.is_none());
                    reflected_vertex_inputs_attribute_descriptions = Some((
                        shader_reflection.inputs_stride,
                        shader_reflection.inputs.clone(),
                    ));
                }

                module_data.push((shader_module, shader_reflection));
            }

            (module_data, shader_stage_create_info)
        };

        let pipeline_data = if let Some(layout) = options.layout {
            PipelineData::Reference { layout }
        } else {
            let mut descriptor_set_table =
                std::collections::HashMap::<u32, DescriptorSetLayoutBinding>::new();

            let mut push_constant_ranges = Vec::<PushConstantRange>::new();

            for (_, reflected_shader) in shader_modules_data.iter() {
                for (&set_id, set_bindings) in reflected_shader.descriptor_sets.iter() {
                    if set_bindings.is_empty() {
                        //
                        // not used in this stage, ignore it
                        continue;
                    }

                    descriptor_set_table
                        .entry(set_id)
                        .and_modify(|e| {
                            if e.descriptor_type == set_bindings[0].descriptor_type {
                                e.stage_flags |= set_bindings[0].stage_flags;
                                e.descriptor_count = e.descriptor_count.max(set_bindings[0].descriptor_count);
                            } else {
                                panic!("Sets alias to the same slot {set_id} but descriptor types are not compatible {:?}/{:?}",
                                        e.descriptor_type, set_bindings[0].descriptor_type);
                            }
                        })
                        .or_insert( DescriptorSetLayoutBinding {
                            descriptor_count: 1024,
                            ..set_bindings[0]
                        });
                }

                push_constant_ranges.extend_from_slice(&reflected_shader.push_constants);
            }

            let compare_push_consts = |r0: &PushConstantRange, r1: &PushConstantRange| {
                if r0.stage_flags != r1.stage_flags {
                    return r0.stage_flags.cmp(&r1.stage_flags);
                }

                if r0.offset != r1.offset {
                    return r0.offset.cmp(&r1.offset);
                }

                r0.size.cmp(&r1.size)
            };

            push_constant_ranges.sort_by(compare_push_consts);
            push_constant_ranges
                .dedup_by(|r0, r1| compare_push_consts(r0, r1) == std::cmp::Ordering::Equal);

            log::info!("pipeline layout definition {descriptor_set_table:?}");
            log::info!("pipeline push constant ranges {push_constant_ranges:?}");

            let max_set_id = descriptor_set_table
                .iter()
                .map(|(set_id, _)| *set_id)
                .max()
                .unwrap_or(0);

            //
            // Plug descriptor set layout holes.
            // For example a vertex shader might have layout (set = 0, binding = ...)
            // and the fragment shader might have layout (set = 4, binding = ...)
            // For sets 1 to 3 we need to create a "null" descriptor set layout with no bindings
            // and assign it to those slots when creating the pipeline layout.

            let descriptor_set_layout = (0..=max_set_id)
                .map(|set_id| {
                    if let Some(set_entry) = descriptor_set_table.get(&set_id) {
                        let binding_flags = [DescriptorBindingFlags::PARTIALLY_BOUND];
                        let mut set_layout_binding_flags =
                            *DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                                .binding_flags(&binding_flags);

                        unsafe {
                            renderer
                                .logical()
                                .create_descriptor_set_layout(
                                    &DescriptorSetLayoutCreateInfo::builder()
                                        .push_next(&mut set_layout_binding_flags)
                                        .bindings(std::slice::from_raw_parts(
                                            set_entry as *const _,
                                            1,
                                        )),
                                    None,
                                )
                                .expect("Monka mega, need a better way to deal with failure here")
                        }
                    } else {
                        renderer.empty_descriptor_set_layout()
                    }
                })
                .collect::<Vec<_>>();

            PipelineData::Owned {
                layout: unsafe {
                    renderer.logical().create_pipeline_layout(
                        &PipelineLayoutCreateInfo::builder()
                            .set_layouts(&descriptor_set_layout)
                            .push_constant_ranges(&push_constant_ranges),
                        None,
                    )
                }?,
                descriptor_set_layout,
            }
        };

        let mut render_state_create_info = match &renderer.renderstate {
            RenderState::Dynamic {
                color_attachments,
                depth_attachments,
                stencil_attachments,
            } => *PipelineRenderingCreateInfo::builder()
                .color_attachment_formats(color_attachments)
                .depth_attachment_format(depth_attachments[0])
                .stencil_attachment_format(stencil_attachments[0]),
            RenderState::Renderpass(_) => *PipelineRenderingCreateInfo::builder(),
        };

        let input_assembly_state = *ash::vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(ash::vk::PrimitiveTopology::TRIANGLE_LIST);

        let (inputs_stride, input_rate) = self
            .input_assembly_state
            .as_ref()
            .map(|ia| (ia.stride, ia.input_rate))
            .unwrap_or_else(|| {
                if let Some((stride, _)) = reflected_vertex_inputs_attribute_descriptions.as_ref() {
                    (*stride, VertexInputRate::VERTEX)
                } else {
                    (0, VertexInputRate::VERTEX)
                }
            });

        let std_vertex_binding = [*ash::vk::VertexInputBindingDescription::builder()
            .stride(inputs_stride)
            .input_rate(input_rate)
            .binding(0)];

        let vertex_input_state = self
            .input_assembly_state
            .as_ref()
            .map(|ia| {
                *PipelineVertexInputStateCreateInfo::builder()
                    .vertex_attribute_descriptions(&ia.vertex_descriptions)
                    .vertex_binding_descriptions(&std_vertex_binding)
            })
            .unwrap_or_else(|| {
                reflected_vertex_inputs_attribute_descriptions
                    .as_ref()
                    .map_or_else(
                        || *PipelineVertexInputStateCreateInfo::builder(),
                        |(_, vertex_inputs)| {
                            *PipelineVertexInputStateCreateInfo::builder()
                                .vertex_attribute_descriptions(vertex_inputs)
                                .vertex_binding_descriptions(&std_vertex_binding)
                        },
                    )
            });

        let rasterization_state = self.rasterization_state.unwrap_or_else(|| {
            *PipelineRasterizationStateCreateInfo::builder()
                .polygon_mode(PolygonMode::FILL)
                .cull_mode(CullModeFlags::BACK)
                .front_face(FrontFace::COUNTER_CLOCKWISE)
                .depth_bias_enable(false)
                .line_width(1f32)
        });

        let multisample_state = self.multisample_state.unwrap_or_else(|| {
            *PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(SampleCountFlags::TYPE_1)
        });

        let depth_stencil_state = self.depth_stencil_state.unwrap_or_else(|| {
            *PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(CompareOp::LESS)
                .depth_bounds_test_enable(true)
                .min_depth_bounds(0f32)
                .max_depth_bounds(1f32)
        });

        let colorblend_attachments = [*ash::vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(false)
            .src_color_blend_factor(BlendFactor::ONE)
            .dst_color_blend_factor(BlendFactor::ZERO)
            .color_blend_op(BlendOp::ADD)
            .src_alpha_blend_factor(BlendFactor::ONE)
            .dst_alpha_blend_factor(BlendFactor::ZERO)
            .alpha_blend_op(BlendOp::ADD)
            .color_write_mask(
                ColorComponentFlags::R
                    | ColorComponentFlags::G
                    | ColorComponentFlags::B
                    | ColorComponentFlags::A,
            )];

        let colorblend_state = self.color_blend_state.unwrap_or_else(|| {
            *PipelineColorBlendStateCreateInfo::builder()
                .blend_constants([1f32; 4])
                .attachments(&colorblend_attachments)
        });

        let dynamic_state =
            PipelineDynamicStateCreateInfo::builder().dynamic_states(&self.dynamic_state);

        let viewport_state = *ash::vk::PipelineViewportStateCreateInfo::builder();

        let graphics_pipeline_create_info = {
            let mut pipeline_create_info = GraphicsPipelineCreateInfo::builder()
                .input_assembly_state(&input_assembly_state)
                .stages(&pipeline_shader_stage_create_info)
                .vertex_input_state(&vertex_input_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
                .depth_stencil_state(&depth_stencil_state)
                .color_blend_state(&colorblend_state)
                .viewport_state(&viewport_state)
                .dynamic_state(&dynamic_state)
                .layout(pipeline_data.layout());

            match renderer.renderstate {
                RenderState::Renderpass(pass) => {
                    pipeline_create_info = pipeline_create_info.render_pass(pass).subpass(0);
                }
                RenderState::Dynamic { .. } => {
                    pipeline_create_info =
                        pipeline_create_info.push_next(&mut render_state_create_info);
                }
            }

            *pipeline_create_info
        };

        let pipeline_handles = unsafe {
            renderer.logical().create_graphics_pipelines(
                renderer.pipeline_cache(),
                std::slice::from_raw_parts(&graphics_pipeline_create_info as *const _, 1),
                None,
            )
        }
        .map_err(|(_, e)| GraphicsError::VulkanApi(e))?;

        Ok(UniquePipeline {
            device: renderer.logical_raw(),
            handle: pipeline_handles[0],
            data: pipeline_data,
        })
    }
}

pub mod misc {
    pub fn write_ppm<P: AsRef<std::path::Path>>(file: P, width: u32, height: u32, pixels: &[u8]) {
        use std::io::Write;
        let mut f = std::fs::File::create(file).unwrap();

        writeln!(&mut f, "P3 {width} {height} 255").unwrap();
        pixels.chunks(4).for_each(|c| {
            writeln!(&mut f, "{} {} {}", c[0], c[1], c[2]).unwrap();
        });
    }
}
