use ash::vk::{
    AccessFlags2, BufferImageCopy, CommandBuffer, DependencyFlags, DependencyInfo, DeviceMemory,
    Format, Image, ImageAspectFlags, ImageCreateInfo, ImageLayout, ImageMemoryBarrier2,
    ImageSubresourceLayers, ImageSubresourceRange, ImageType, ImageUsageFlags, ImageView,
    ImageViewCreateInfo, ImageViewType, MemoryPropertyFlags, PipelineStageFlags2,
};

use crate::vulkan_renderer::{GraphicsError, QueueType, QueuedJob, VulkanRenderer};

#[derive(Copy, Clone, Default, Debug, Hash, Eq, PartialEq)]
pub struct VulkanTextureInfo {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub image_layout: ash::vk::ImageLayout,
    pub image_format: ash::vk::Format,
    pub level_count: u32,
    pub layer_count: u32,
    pub view_type: ash::vk::ImageViewType,
}

pub struct VulkanImageCreateInfo<'a> {
    pub tag_name: Option<&'a str>,
    pub work_pkg: Option<&'a QueuedJob>,
    pub ty: ImageType,
    pub usage: ImageUsageFlags,
    pub memory: MemoryPropertyFlags,
    pub format: Format,
    pub cubemap: bool,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub layers: u32,
    pub pixels: &'a [&'a [u8]],
}

pub struct UniqueImage {
    device: *const ash::Device,
    pub image: Image,
    pub memory: DeviceMemory,
    pub view: ImageView,
    pub info: VulkanTextureInfo,
}

impl UniqueImage {
    pub fn from_bytes(
        renderer: &mut VulkanRenderer,
        create_info: &VulkanImageCreateInfo,
    ) -> std::result::Result<UniqueImage, GraphicsError> {
        let image_view_type = match create_info.ty {
            ImageType::TYPE_1D => {
                if create_info.layers > 1 {
                    ImageViewType::TYPE_1D_ARRAY
                } else {
                    ImageViewType::TYPE_1D
                }
            }
            ImageType::TYPE_2D => {
                if create_info.cubemap {
                    if create_info.layers > 1 {
                        ImageViewType::CUBE_ARRAY
                    } else {
                        ImageViewType::CUBE
                    }
                } else {
                    if create_info.layers > 1 {
                        ImageViewType::TYPE_2D_ARRAY
                    } else {
                        ImageViewType::TYPE_2D
                    }
                }
            }
            ImageType::TYPE_3D => ImageViewType::TYPE_3D,
            _ => {
                panic!("Unhandled image type {:?}", create_info.ty);
            }
        };

        let image = unsafe {
            renderer.logical().create_image(
                &ImageCreateInfo::default()
                    .image_type(create_info.ty)
                    .format(create_info.format)
                    .extent(ash::vk::Extent3D {
                        width: create_info.width,
                        height: create_info.height,
                        depth: create_info.depth,
                    })
                    .mip_levels(1)
                    .array_layers(create_info.layers)
                    .samples(ash::vk::SampleCountFlags::TYPE_1)
                    .tiling(ash::vk::ImageTiling::OPTIMAL)
                    .usage(create_info.usage | ImageUsageFlags::TRANSFER_DST)
                    .sharing_mode(ash::vk::SharingMode::EXCLUSIVE)
                    .initial_layout(ImageLayout::UNDEFINED),
                None,
            )
        }?;

        scopeguard::defer_on_unwind! {
            unsafe {
                renderer.logical().destroy_image(image, None);
            }
        }

        let memory_req = unsafe { renderer.logical().get_image_memory_requirements(image) };
        let image_memory = unsafe {
            renderer.logical().allocate_memory(
                &ash::vk::MemoryAllocateInfo::default()
                    .allocation_size(memory_req.size)
                    .memory_type_index(
                        renderer.choose_memory_heap(&memory_req, MemoryPropertyFlags::DEVICE_LOCAL),
                    ),
                None,
            )
        }?;

        scopeguard::defer_on_unwind! {
            unsafe {
                renderer.logical().free_memory(image_memory, None);
            }
        }

        unsafe {
            renderer
                .logical()
                .bind_image_memory(image, image_memory, 0)?;
        }

        create_info
            .tag_name
            .map(|tag_name| renderer.debug_set_object_name(image, tag_name));

        let maybe_pixels = if create_info.pixels.is_empty() {
            Some(())
        } else {
            None
        };

        maybe_pixels.and(create_info.work_pkg).map(|queued_job| {
            let (mut staging_ptr, staging_buffer, staging_offset) =
                renderer.reserve_staging_memory(memory_req.size as usize);

            for p in create_info.pixels {
                unsafe {
                    std::ptr::copy_nonoverlapping(p.as_ptr(), staging_ptr, p.len());
                    staging_ptr = staging_ptr.offset(p.len() as isize);
                }
            }

            let image_subresource_range = ImageSubresourceRange::default()
                .aspect_mask(ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(create_info.layers);

            //
            // set image layout to TRANSFER_DST_OPTIMAL
            set_image_layout(
                queued_job.cmd_buffer,
                renderer.logical(),
                image,
                ImageLayout::UNDEFINED,
                ImageLayout::TRANSFER_DST_OPTIMAL,
                image_subresource_range,
            );

            unsafe {
                renderer.logical().cmd_copy_buffer_to_image(
                    queued_job.cmd_buffer,
                    staging_buffer,
                    image,
                    ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[BufferImageCopy::default()
                        .buffer_offset(staging_offset as u64)
                        .image_subresource(
                            ImageSubresourceLayers::default()
                                .aspect_mask(ImageAspectFlags::COLOR)
                                .mip_level(0)
                                .base_array_layer(0)
                                .layer_count(create_info.layers),
                        )
                        .image_extent(ash::vk::Extent3D {
                            width: create_info.width,
                            height: create_info.height,
                            depth: create_info.depth,
                        })],
                );
            }

            let (qgraphics, _, _, _, _) = renderer.queue_data(QueueType::Graphics);
            let (qtransfer, _, _, _, _) = renderer.queue_data(QueueType::Transfer);

            //
            // post copy memory barrier for queue ownership transfer
            unsafe {
                renderer.logical().cmd_pipeline_barrier2(
                    queued_job.cmd_buffer,
                    &DependencyInfo::default()
                        .dependency_flags(DependencyFlags::BY_REGION)
                        .image_memory_barriers(&[ImageMemoryBarrier2::default()
                            .src_access_mask(AccessFlags2::TRANSFER_WRITE)
                            .src_stage_mask(PipelineStageFlags2::TRANSFER)
                            .src_queue_family_index(qtransfer)
                            .dst_queue_family_index(qgraphics)
                            .image(image)
                            .old_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
                            .new_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .subresource_range(image_subresource_range)]),
                )
            }
        });

        let image_view = unsafe {
            renderer.logical().create_image_view(
                &ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(image_view_type)
                    .format(create_info.format)
                    .components(ash::vk::ComponentMapping::default())
                    .subresource_range(
                        ImageSubresourceRange::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(create_info.layers),
                    ),
                None,
            )
        }?;

        log::info!("[[VK]] - new image {} : handle {image:p}, memory {image_memory:p}, view {image_view:p}", create_info.tag_name.unwrap_or_else(|| "anonymous"));
        Ok(UniqueImage {
            device: renderer.logical_raw(),
            image,
            memory: image_memory,
            view: image_view,
            info: VulkanTextureInfo {
                width: create_info.width,
                height: create_info.height,
                depth: create_info.depth,
                image_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image_format: create_info.format,
                level_count: 1,
                layer_count: create_info.layers,
                view_type: image_view_type,
            },
        })
    }
}

impl std::ops::Drop for UniqueImage {
    fn drop(&mut self) {
        unsafe {
            (*self.device).destroy_image(self.image, None);
            (*self.device).free_memory(self.memory, None);
            (*self.device).destroy_image_view(self.view, None);
        }
    }
}

pub fn set_image_layout(
    cmd_buffer: ash::vk::CommandBuffer,
    device: &ash::Device,
    image: ash::vk::Image,
    initial_layout: ash::vk::ImageLayout,
    final_layout: ash::vk::ImageLayout,
    subresource_range: ash::vk::ImageSubresourceRange,
) {
    unsafe {
        device.cmd_pipeline_barrier2(
            cmd_buffer,
            &ash::vk::DependencyInfo::default()
                .dependency_flags(ash::vk::DependencyFlags::BY_REGION)
                .image_memory_barriers(&[image_layout_memory_barrier(
                    image,
                    initial_layout,
                    final_layout,
                    subresource_range,
                )]),
        );
    }
}

fn image_layout_memory_barrier<'a>(
    image: ash::vk::Image,
    previous_layout: ash::vk::ImageLayout,
    new_layout: ash::vk::ImageLayout,
    subresource_range: ash::vk::ImageSubresourceRange,
) -> ash::vk::ImageMemoryBarrier2<'a> {
    let mut mem_barrier = ImageMemoryBarrier2::default()
        .src_stage_mask(PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(AccessFlags2::empty())
        .dst_stage_mask(PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(AccessFlags2::empty())
        .old_layout(previous_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(subresource_range);

    //
    // Source layouts (old)
    // The source access mask controls actions to be finished on the old
    // layout before it will be transitioned to the new layout.
    match previous_layout {
        ImageLayout::UNDEFINED => {
            // Image layout is undefined (or does not matter).
            // Only valid as initial layout. No flags required.
            // Do nothing.
        }

        ImageLayout::PREINITIALIZED => {
            // Image is preinitialized.
            // Only valid as initial layout for linear images; preserves memory
            // contents. Make sure host writes have finished.
            mem_barrier.src_access_mask = AccessFlags2::HOST_WRITE;
        }

        ImageLayout::COLOR_ATTACHMENT_OPTIMAL => {
            // Image is a color attachment.
            // Make sure writes to the color buffer have finished
            mem_barrier.src_access_mask = AccessFlags2::COLOR_ATTACHMENT_WRITE;
        }

        ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
            // Image is a depth/stencil attachment.
            // Make sure any writes to the depth/stencil buffer have finished.
            mem_barrier.src_access_mask = AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
        }

        ImageLayout::TRANSFER_SRC_OPTIMAL => {
            // Image is a transfer source.
            // Make sure any reads from the image have finished
            mem_barrier.src_access_mask = AccessFlags2::TRANSFER_READ;
        }

        ImageLayout::TRANSFER_DST_OPTIMAL => {
            // Image is a transfer destination.
            // Make sure any writes to the image have finished.
            mem_barrier.src_access_mask = AccessFlags2::TRANSFER_WRITE;
        }

        ImageLayout::SHADER_READ_ONLY_OPTIMAL => {
            // Image is read by a shader.
            // Make sure any shader reads from the image have finished
            mem_barrier.src_access_mask = AccessFlags2::SHADER_READ;
        }

        _ => {
            // Value not used by callers, so not supported.
            log::error!(
                "Unsupported value for previous layoutr of image: {:?}",
                previous_layout
            );
        }
    }

    // Target layouts (new)
    // The destination access mask controls the dependency for the new image
    // layout.
    match new_layout {
        ImageLayout::TRANSFER_DST_OPTIMAL => {
            // Image will be used as a transfer destination.
            // Make sure any writes to the image have finished.
            mem_barrier.dst_access_mask = AccessFlags2::TRANSFER_WRITE;
        }

        ImageLayout::TRANSFER_SRC_OPTIMAL => {
            // Image will be used as a transfer source.
            // Make sure any reads from and writes to the image have finished.
            mem_barrier.src_access_mask |= AccessFlags2::TRANSFER_READ;
            mem_barrier.dst_access_mask = AccessFlags2::TRANSFER_READ;
        }

        ImageLayout::COLOR_ATTACHMENT_OPTIMAL => {
            // Image will be used as a color attachment.
            // Make sure any writes to the color buffer have finished.
            mem_barrier.src_access_mask = AccessFlags2::TRANSFER_READ;
            mem_barrier.dst_access_mask = AccessFlags2::COLOR_ATTACHMENT_WRITE;
        }

        ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
            // Image layout will be used as a depth/stencil attachment.
            // Make sure any writes to depth/stencil buffer have finished.
            mem_barrier.dst_access_mask = AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
        }

        ImageLayout::SHADER_READ_ONLY_OPTIMAL => {
            // Image will be read in a shader (sampler, input attachment).
            // Make sure any writes to the image have finished.
            if mem_barrier.src_access_mask.is_empty() {
                mem_barrier.src_access_mask =
                    AccessFlags2::HOST_WRITE | AccessFlags2::TRANSFER_WRITE;
            }
            mem_barrier.dst_access_mask = AccessFlags2::SHADER_READ;
        }
        _ => {
            // Value not used by callers, so not supported.
            log::error!("Unsupported value for new layout of image: {new_layout:?}");
        }
    }

    mem_barrier
}

pub struct UniqueSampler {
    device: *const ash::Device,
    pub handle: ash::vk::Sampler,
}

impl std::ops::Drop for UniqueSampler {
    fn drop(&mut self) {
        unsafe {
            (*self.device).destroy_sampler(self.handle, None);
        }
    }
}

impl UniqueSampler {
    pub fn new(
        vks: &VulkanRenderer,
        create_info: ash::vk::SamplerCreateInfo,
    ) -> std::result::Result<Self, GraphicsError> {
        let handle = unsafe { vks.logical().create_sampler(&create_info, None) }?;

        Ok(UniqueSampler {
            device: vks.logical_raw(),
            handle,
        })
    }
}
