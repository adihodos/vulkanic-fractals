use ash::{
    vk::{
        DescriptorSetLayout, DescriptorSetLayoutCreateInfo, DeviceMemory,
        GraphicsPipelineCreateInfo, Image, ImageCreateInfo, ImageView, ImageViewCreateInfo,
        MemoryAllocateInfo, MemoryPropertyFlags, MemoryRequirements, Pipeline, PipelineCache,
        PipelineLayout, PipelineLayoutCreateInfo, Sampler, SamplerCreateInfo, ShaderModule,
        ShaderModuleCreateInfo,
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

    let src_code = std::fs::read_to_string(&p).expect("Failed to read shader file {}");

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
