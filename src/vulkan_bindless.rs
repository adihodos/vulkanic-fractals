use std::sync::atomic::AtomicU32;

use ash::vk::{
    DescriptorBufferInfo, DescriptorImageInfo, DescriptorPool, DescriptorPoolCreateInfo,
    DescriptorPoolSize, DescriptorSet, DescriptorSetLayout, DescriptorType, PipelineLayout,
    WriteDescriptorSet,
};
use smallvec::SmallVec;

use crate::{
    vulkan_buffer::VulkanBuffer,
    vulkan_image::{UniqueImage, UniqueSampler, VulkanTextureInfo},
    vulkan_renderer::{GraphicsError, VulkanRenderer},
};

/// Encodes some data for the GPU as a shader push constant (u32).
/// bits [0..3] frame id
/// bits [4..19] bindless resource index
/// bits [20..31] not used atm
#[derive(Copy, Clone)]
pub struct GlobalPushConstant(u32);

impl GlobalPushConstant {
    pub fn from_resource<T>(
        resource: BindlessResourceHandleCore<T>,
        frame_id: u32,
        misc: Option<u32>,
    ) -> Self {
        assert!(frame_id < 16);
        assert!(misc.unwrap_or_default() < (1 << 12));

        let resource_handle = resource.element_handle(frame_id as usize).handle();
        assert!(resource_handle < (1 << 20));

        Self((misc.unwrap_or_default() << 20) | (resource_handle << 4) | frame_id)
    }

    pub fn to_gpu(&self) -> [u8; 4] {
        self.0.to_le_bytes()
    }
}

impl std::fmt::Debug for GlobalPushConstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GlobalPushConstant value: {0} - {0:b} -> [misc {1}, resource: {2}, frame: {3}]",
            self.0,
            self.0 >> 20,
            (self.0 >> 4) & 0xFFFF,
            self.0 & 0xF
        )
    }
}

impl std::convert::AsRef<[u8]> for GlobalPushConstant {
    fn as_ref(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(&self.0 as *const _ as *const u8, std::mem::size_of::<u32>())
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct BindlessResourceHandleCore<T> {
    id: u32,
    _marker: std::marker::PhantomData<T>,
}

impl<T> std::fmt::Debug for BindlessResourceHandleCore<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BindlessResourceHandle value: {} -> [handle {}, array elements {}]",
            self.id,
            self.handle(),
            self.array_elements()
        )
    }
}

impl<T> BindlessResourceHandleCore<T> {
    const HANDLE_MASK: u32 = 0b000000000000_1111_1111_1111_1111_1111;
    const ELEMENT_MASK: u32 = 0b00000000000000000000_1111_1111_1111;

    pub fn new(resource: u32, array_elements: u32) -> Self {
        assert!(resource < 1048576);
        assert!(array_elements < 4096);
        Self {
            id: (resource & Self::HANDLE_MASK) | (array_elements << 20),
            _marker: std::marker::PhantomData {},
        }
    }

    pub fn handle(&self) -> u32 {
        self.id & Self::HANDLE_MASK
    }

    pub fn array_elements(&self) -> u32 {
        (self.id >> 20) & Self::ELEMENT_MASK
    }

    pub fn element_handle(&self, index: usize) -> Self {
        let elements = self.array_elements();
        assert!(index < elements as usize);

        Self::new(self.handle() + index as u32, 1)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct TagUniformBufferResource {}
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct TagStorageBufferResource {}
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct TagImageResource {}

pub type BindlessUniformBufferHandle = BindlessResourceHandleCore<TagUniformBufferResource>;
pub type BindlessStoragebufferHandle = BindlessResourceHandleCore<TagStorageBufferResource>;
pub type BindlessImageHandle = BindlessResourceHandleCore<TagImageResource>;

/// Either SBO or UBO
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct BindlessResourceEntryBuffer {
    pub buffer: ash::vk::Buffer,
    pub devmem: ash::vk::DeviceMemory,
    pub aligned_slab_size: usize,
}

impl std::default::Default for BindlessResourceEntryBuffer {
    fn default() -> Self {
        Self {
            buffer: ash::vk::Buffer::null(),
            devmem: ash::vk::DeviceMemory::null(),
            aligned_slab_size: 0,
        }
    }
}

/// Image + view
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct BindlessResourceEntryImage {
    pub image: ash::vk::Image,
    pub devmem: ash::vk::DeviceMemory,
    pub view: ash::vk::ImageView,
    pub info: VulkanTextureInfo,
}

impl std::default::Default for BindlessResourceEntryImage {
    fn default() -> Self {
        Self {
            image: ash::vk::Image::null(),
            devmem: ash::vk::DeviceMemory::null(),
            view: ash::vk::ImageView::null(),
            info: VulkanTextureInfo::default(),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct BindlessUniformBufferResourceHandleEntryPair(
    pub BindlessUniformBufferHandle,
    pub BindlessResourceEntryBuffer,
);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct BindlessStorageBufferResourceHandleEntryPair(
    pub BindlessStoragebufferHandle,
    pub BindlessResourceEntryBuffer,
);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct BindlessImageResourceHandleEntryPair(
    pub BindlessImageHandle,
    pub BindlessResourceEntryImage,
);

pub struct BindlessResourceSystem {
    device: *const ash::Device,
    dpool: DescriptorPool,
    set_layouts: Vec<DescriptorSetLayout>,
    descriptor_sets: Vec<DescriptorSet>,
    bindless_pipeline_layout: PipelineLayout,

    ubo_pending_writes: Vec<(u32, DescriptorBufferInfo)>,
    ssbo_pending_writes: Vec<(u32, DescriptorBufferInfo)>,
    cso_pending_writes: Vec<(u32, DescriptorImageInfo)>,

    storage_buffers: Vec<BindlessResourceEntryBuffer>,
    uniform_buffers: Vec<BindlessResourceEntryBuffer>,
    combined_samplers: Vec<BindlessResourceEntryImage>,

    sbo_free_slot: std::sync::atomic::AtomicU32,
    ubo_free_slot: std::sync::atomic::AtomicU32,
    cso_free_slot: std::sync::atomic::AtomicU32,
}

impl std::ops::Drop for BindlessResourceSystem {
    fn drop(&mut self) {
        let free_buffer_res = |buffer_resource: &BindlessResourceEntryBuffer| unsafe {
            (*self.device).free_memory(buffer_resource.devmem, None);
            (*self.device).destroy_buffer(buffer_resource.buffer, None);
        };
        self.storage_buffers
            .iter()
            .for_each(|bindless_buffer| free_buffer_res(bindless_buffer));
        self.uniform_buffers
            .iter()
            .for_each(|uniform_buffer| free_buffer_res(uniform_buffer));
        self.combined_samplers
            .iter()
            .for_each(|combined_sampler| unsafe {
                (*self.device).free_memory(combined_sampler.devmem, None);
                (*self.device).destroy_image_view(combined_sampler.view, None);
                (*self.device).destroy_image(combined_sampler.image, None);
            });

        unsafe {
            self.set_layouts.iter().for_each(|&set_layout| {
                (*self.device).destroy_descriptor_set_layout(set_layout, None);
            });
            (*self.device).destroy_descriptor_pool(self.dpool, None);
            (*self.device).destroy_pipeline_layout(self.bindless_pipeline_layout, None);
        }
    }
}

impl BindlessResourceSystem {
    const BINDLESS_RESOURCE_INDEX_UBO: usize = 0;
    const BINDLESS_RESOURCE_INDEX_SBO: usize = 1;
    const BINDLESS_RESOURCE_INDEX_CSO: usize = 2;

    const DESCRIPTOR_POOL_SIZES: [u32; 3] = [128, 1024, 1024];
    const DESCRIPTOR_TYPES: [DescriptorType; 3] = [
        DescriptorType::UNIFORM_BUFFER,
        DescriptorType::STORAGE_BUFFER,
        DescriptorType::COMBINED_IMAGE_SAMPLER,
    ];

    pub fn descriptor_sets(&self) -> &[DescriptorSet] {
        &self.descriptor_sets
    }
    pub fn pipeline_layout(&self) -> PipelineLayout {
        self.bindless_pipeline_layout
    }

    pub fn new(vks: &VulkanRenderer) -> Result<Self, GraphicsError> {
        let dpool_sizes = Self::DESCRIPTOR_TYPES
            .iter()
            .zip(Self::DESCRIPTOR_POOL_SIZES.iter())
            .map(|(&desc_type, &pool_size)| {
                DescriptorPoolSize::default()
                    .ty(desc_type)
                    .descriptor_count(pool_size)
            })
            .collect::<SmallVec<[DescriptorPoolSize; 3]>>();

        let dpool = unsafe {
            vks.logical().create_descriptor_pool(
                &DescriptorPoolCreateInfo::default()
                    .flags(ash::vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                    .pool_sizes(&dpool_sizes)
                    .max_sets(8192),
                None,
            )
        }?;

        let device = vks.logical_raw();

        scopeguard::defer_on_unwind! {
            unsafe {
                (*device).destroy_descriptor_pool(dpool, None);
            }
        }

        let set_layouts: Vec<DescriptorSetLayout> = std::result::Result::from_iter(
            [
                (
                    DescriptorType::UNIFORM_BUFFER,
                    vks.properties()
                        .descriptor_indexing
                        .max_per_stage_descriptor_update_after_bind_uniform_buffers
                        .min(64),
                    ash::vk::DescriptorBindingFlags::PARTIALLY_BOUND,
                    "Bindless - set layout UB",
                ),
                (
                    DescriptorType::STORAGE_BUFFER,
                    1024,
                    ash::vk::DescriptorBindingFlags::PARTIALLY_BOUND
                        | ash::vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                    "Bindless - set layout SB",
                ),
                (
                    DescriptorType::COMBINED_IMAGE_SAMPLER,
                    1024,
                    ash::vk::DescriptorBindingFlags::PARTIALLY_BOUND
                        | ash::vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                    "Bindless - set layout CS",
                ),
            ]
            .iter()
            .map(
                |&(descriptor_type, descriptor_count, binding_flags, tag)| unsafe {
                    log::info!("Bindless:: {descriptor_type:?}, {descriptor_count}");
                    let mut flag_info =
                        ash::vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                            .binding_flags(std::slice::from_ref(&binding_flags));

                    vks.logical()
                        .create_descriptor_set_layout(
                            &ash::vk::DescriptorSetLayoutCreateInfo::default()
                                .push_next(&mut flag_info)
                                .flags(
                                    ash::vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
                                )
                                .bindings(&[ash::vk::DescriptorSetLayoutBinding::default()
                                    .descriptor_count(descriptor_count)
                                    .binding(0)
                                    .descriptor_type(descriptor_type)
                                    .stage_flags(ash::vk::ShaderStageFlags::ALL)]),
                            None,
                        )
                        .and_then(|set_layout| {
                            vks.debug_set_object_name(set_layout, tag);
                            Ok(set_layout)
                        })
                },
            )
            .collect::<Vec<_>>(),
        )?;

        let descriptor_sets = unsafe {
            vks.logical().allocate_descriptor_sets(
                &ash::vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(dpool)
                    .set_layouts(&set_layouts),
            )
        }?;

        let bindless_pipeline_layout = unsafe {
            vks.logical().create_pipeline_layout(
                &ash::vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&[ash::vk::PushConstantRange::default()
                        .stage_flags(ash::vk::ShaderStageFlags::ALL)
                        .offset(0)
                        .size(size_of::<u32>() as _)]),
                None,
            )
        }?;

        #[cfg(debug_assertions)]
        {
            vks.debug_set_object_name(dpool, "Bindless descriptor pool");
            vks.debug_set_object_name(bindless_pipeline_layout, "Bindless pipeline layout");
            Self::DESCRIPTOR_TYPES
                .iter()
                .zip(descriptor_sets.iter())
                .for_each(|(&descriptor_type, &descriptor_set)| {
                    vks.debug_set_object_name(
                        descriptor_set,
                        &format!("Bindless DS {descriptor_type:?}"),
                    );
                });
        }

        Ok(Self {
            device: vks.logical_raw(),
            dpool,
            set_layouts,
            descriptor_sets,
            bindless_pipeline_layout,
            ssbo_pending_writes: vec![],
            ubo_pending_writes: vec![],
            cso_pending_writes: vec![],

            storage_buffers: vec![],
            uniform_buffers: vec![],
            combined_samplers: vec![],

            sbo_free_slot: AtomicU32::new(0),
            ubo_free_slot: AtomicU32::new(0),
            cso_free_slot: AtomicU32::new(0),
        })
    }

    pub fn register_uniform_buffer(
        &mut self,
        ubo: VulkanBuffer,
        slot: Option<u32>,
    ) -> BindlessUniformBufferResourceHandleEntryPair {
        let global_entry = slot.unwrap_or_else(|| {
            self.ubo_free_slot
                .fetch_add(ubo.slabs as u32, std::sync::atomic::Ordering::Acquire)
        });

        if global_entry >= self.uniform_buffers.len() as u32 {
            self.uniform_buffers.resize(
                global_entry as usize + 1,
                BindlessResourceEntryBuffer::default(),
            );
        }

        let bindless_handle = BindlessUniformBufferHandle::new(global_entry, ubo.slabs as u32);

        self.ubo_pending_writes.extend((0..ubo.slabs).map(|i| {
            (
                global_entry + i as u32,
                DescriptorBufferInfo::default()
                    .buffer(ubo.buffer)
                    .offset((ubo.aligned_slab_size * i) as u64)
                    .range(ubo.aligned_slab_size as u64),
            )
        }));

        let ubo_entry_data = BindlessResourceEntryBuffer {
            buffer: ubo.buffer,
            devmem: ubo.memory,
            aligned_slab_size: ubo.aligned_slab_size,
        };

        self.uniform_buffers[global_entry as usize] = ubo_entry_data;
        std::mem::forget(ubo);
        BindlessUniformBufferResourceHandleEntryPair(bindless_handle, ubo_entry_data)
    }

    pub fn register_storage_buffer(
        &mut self,
        sbo: VulkanBuffer,
        slot: Option<u32>,
    ) -> BindlessStorageBufferResourceHandleEntryPair {
        let global_entry = slot.unwrap_or_else(|| {
            self.sbo_free_slot
                .fetch_add(sbo.slabs as u32, std::sync::atomic::Ordering::Acquire)
        });

        if global_entry >= self.storage_buffers.len() as u32 {
            self.storage_buffers.resize(
                global_entry as usize + 1,
                BindlessResourceEntryBuffer::default(),
            );
        }

        let bindless_handle = BindlessStoragebufferHandle::new(global_entry, sbo.slabs as u32);

        self.ssbo_pending_writes.extend((0..sbo.slabs).map(|i| {
            (
                global_entry + i as u32,
                DescriptorBufferInfo::default()
                    .buffer(sbo.buffer)
                    .offset((i * sbo.aligned_slab_size) as u64)
                    .range(sbo.aligned_slab_size as u64),
            )
        }));

        let sbo_entry_data = BindlessResourceEntryBuffer {
            buffer: sbo.buffer,
            devmem: sbo.memory,
            aligned_slab_size: sbo.aligned_slab_size,
        };

        self.storage_buffers[global_entry as usize] = sbo_entry_data;
        std::mem::forget(sbo);
        BindlessStorageBufferResourceHandleEntryPair(bindless_handle, sbo_entry_data)
    }

    pub fn register_image(
        &mut self,
        image: UniqueImage,
        sampler: &UniqueSampler,
        slot: Option<u32>,
    ) -> BindlessImageResourceHandleEntryPair {
        let global_entry = slot.unwrap_or_else(|| {
            self.cso_free_slot
                .fetch_add(1u32, std::sync::atomic::Ordering::Acquire)
        });

        if global_entry >= self.combined_samplers.len() as u32 {
            self.combined_samplers.resize(
                global_entry as usize + 1,
                BindlessResourceEntryImage::default(),
            );
        }

        let bindless_handle = BindlessImageHandle::new(global_entry, 1);
        self.cso_pending_writes.push((
            global_entry,
            DescriptorImageInfo::default()
                .sampler(sampler.handle)
                .image_view(image.view)
                .image_layout(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        ));

        let cso_entry_data = BindlessResourceEntryImage {
            image: image.image,
            devmem: image.memory,
            view: image.view,
            info: image.info,
        };

        self.combined_samplers[global_entry as usize] = cso_entry_data;
        std::mem::forget(image);

        BindlessImageResourceHandleEntryPair(bindless_handle, cso_entry_data)
    }

    pub fn flush_pending_updates(&mut self, vks: &VulkanRenderer) {
        let mut descriptor_writes = Vec::<WriteDescriptorSet>::new();

        descriptor_writes.extend(self.ubo_pending_writes.iter().map(|pwrite| {
            let s = std::slice::from_ref(&pwrite.1);
            WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[Self::BINDLESS_RESOURCE_INDEX_UBO])
                .dst_binding(0)
                .descriptor_count(1)
                .dst_array_element(pwrite.0)
                .descriptor_type(DescriptorType::UNIFORM_BUFFER)
                .buffer_info(s)
        }));

        descriptor_writes.extend(self.ssbo_pending_writes.iter().map(|pwrite| {
            let s = std::slice::from_ref(&pwrite.1);
            WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[Self::BINDLESS_RESOURCE_INDEX_SBO])
                .dst_binding(0)
                .descriptor_count(1)
                .dst_array_element(pwrite.0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(s)
        }));

        descriptor_writes.extend(self.cso_pending_writes.iter().map(|pwrite| {
            let s = std::slice::from_ref(&pwrite.1);
            WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[Self::BINDLESS_RESOURCE_INDEX_CSO])
                .dst_binding(0)
                .descriptor_count(1)
                .dst_array_element(pwrite.0)
                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(s)
        }));

        unsafe {
            vks.logical()
                .update_descriptor_sets(&descriptor_writes, &[]);

            self.ubo_pending_writes.clear();
            self.ssbo_pending_writes.clear();
            self.cso_pending_writes.clear();
        }
    }

    pub fn bind_descriptors(&self, cmd_buff: ash::vk::CommandBuffer, vks: &VulkanRenderer) {
        unsafe {
            vks.logical().cmd_bind_descriptor_sets(
                cmd_buff,
                ash::vk::PipelineBindPoint::GRAPHICS,
                self.bindless_pipeline_layout,
                0,
                &self.descriptor_sets,
                &[],
            )
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_bindless_handle() {
        use super::*;

        let b0 = BindlessStoragebufferHandle::new(9876, 3456);
        assert_eq!(b0.handle(), 9876);
        assert_eq!(b0.array_elements(), 3456);

        let b1 = b0.element_handle(1000);
        assert_eq!(b1.handle(), 10876);
        assert_eq!(b1.array_elements(), 1);
    }
}
