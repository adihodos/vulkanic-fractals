use std::ffi::c_void;

use ash::{
    vk::{
        Buffer, BufferCreateInfo, BufferUsageFlags, DescriptorSetLayout,
        DescriptorSetLayoutCreateInfo, DeviceMemory, DeviceSize, GraphicsPipelineCreateInfo, Image,
        ImageCreateInfo, ImageView, ImageViewCreateInfo, MappedMemoryRange, MemoryAllocateInfo,
        MemoryMapFlags, MemoryPropertyFlags, MemoryRequirements, Pipeline, PipelineCache,
        PipelineLayout, PipelineLayoutCreateInfo, Sampler, SamplerCreateInfo, ShaderModule,
        ShaderModuleCreateInfo, SharingMode,
    },
    Device,
};

use crate::{VulkanDeviceState, VulkanState};

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
