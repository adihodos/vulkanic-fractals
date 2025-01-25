use ash::vk::{
    BufferCopy, BufferCreateInfo, BufferMemoryRequirementsInfo2, BufferUsageFlags, DeviceSize,
    MemoryAllocateInfo, MemoryPropertyFlags, MemoryRequirements2,
};

use crate::{
    math::round_up,
    vulkan_mapped_memory::UniqueBufferMapping,
    vulkan_renderer::{GraphicsError, QueuedJob, VulkanRenderer},
};

pub struct VulkanBufferCreateInfo<'a> {
    pub name_tag: Option<&'a str>,
    pub work_package: Option<&'a QueuedJob>,
    pub usage: BufferUsageFlags,
    pub memory_properties: MemoryPropertyFlags,
    pub slabs: usize,
    pub bytes: usize,
    pub initial_data: &'a [&'a [u8]],
}

pub struct VulkanBuffer {
    device: *const ash::Device,
    pub buffer: ash::vk::Buffer,
    pub memory: ash::vk::DeviceMemory,
    pub slabs: usize,
    pub aligned_slab_size: usize,
}

impl std::ops::Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            (*self.device).free_memory(self.memory, None);
            (*self.device).destroy_buffer(self.buffer, None);
        }
    }
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

        let alignment_by_memory_access =
            if create_info.memory_properties.intersects(memory_host_access) {
                renderer.limits().non_coherent_atom_size
            } else {
                0
            };

        let alignment_by_usage = {
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

        let alignment = alignment_by_memory_access.max(alignment_by_usage);
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

        let device = renderer.logical_raw();
        let buffer = unsafe {
            (*device).create_buffer(
                &BufferCreateInfo::default()
                    .size(aligned_allocation_size as u64)
                    .usage(usage_flags)
                    .sharing_mode(ash::vk::SharingMode::EXCLUSIVE),
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
            let mut memory_requirements = MemoryRequirements2::default();
            (*device).get_buffer_memory_requirements2(
                &BufferMemoryRequirementsInfo2::default().buffer(buffer),
                &mut memory_requirements,
            );

            (*device).allocate_memory(
                &MemoryAllocateInfo::default()
                    .allocation_size(memory_requirements.memory_requirements.size)
                    .memory_type_index(renderer.choose_memory_heap(
                        &memory_requirements.memory_requirements,
                        create_info.memory_properties,
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
                    let copy_buffer = BufferCopy::default()
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
                    create_info.work_package.unwrap().cmd_buffer,
                    staging_buffer,
                    buffer,
                    &copy_regions,
                );
            }
        }

        let buffer_name = create_info.name_tag.unwrap_or_else(|| "anonymous");

        log::info!(
            "New buffer {buffer_name} @ {buffer:p} <=> {buffer_memory:p}, alignment {alignment},
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
