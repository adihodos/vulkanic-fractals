#![allow(dead_code)]

use crate::vulkan_renderer::{GraphicsError, UniqueShaderModule};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum ShaderKind {
    Vertex,
    Geometry,
    Fragment,
    Compute,
}

impl std::convert::From<ShaderKind> for ash::vk::ShaderStageFlags {
    fn from(value: ShaderKind) -> Self {
        match value {
            ShaderKind::Vertex => ash::vk::ShaderStageFlags::VERTEX,
            ShaderKind::Geometry => ash::vk::ShaderStageFlags::GEOMETRY,
            ShaderKind::Fragment => ash::vk::ShaderStageFlags::FRAGMENT,
            ShaderKind::Compute => ash::vk::ShaderStageFlags::COMPUTE,
        }
    }
}

impl std::convert::TryFrom<&std::path::PathBuf> for ShaderKind {
    type Error = String;

    fn try_from(p: &std::path::PathBuf) -> Result<Self, Self::Error> {
        if let Some(file_ext) = p.extension() {
            match file_ext.to_str() {
                Some("vert") => Ok(ShaderKind::Vertex),
                Some("geom") => Ok(ShaderKind::Geometry),
                Some("frag") => Ok(ShaderKind::Fragment),
                Some("comp") => Ok(ShaderKind::Compute),
                Some(unsupported_ext) => Err(format!(
                    "Unsupported extension {} (must be one of vert, geom, frag, comp)",
                    unsupported_ext
                )),
                None => Err("extension string conversion error".into()),
            }
        } else {
            Err("Missing file extension (must be one of vert, geom, frag, comp)".into())
        }
    }
}

pub enum ShaderSource<'a> {
    File(std::path::PathBuf),
    String(&'a str, ShaderKind),
}

pub struct ShaderCompileInfo<'a> {
    pub src: &'a ShaderSource<'a>,
    pub entry_point: Option<&'a str>,
    pub compile_defs: &'a [(&'a str, Option<&'a str>)],
    pub optimize: bool,
    pub debug_info: bool,
}

fn compile_shader_impl(
    shader_code: &str,
    shader_kind: ShaderKind,
    shader_tag: &str,
    compile_info: &ShaderCompileInfo,
) -> std::result::Result<shaderc::CompilationArtifact, GraphicsError> {
    let mut compiler_options = shaderc::CompileOptions::new()
        .ok_or_else(|| GraphicsError::ShadercGeneric("failed to create CompileOptions".into()))?;

    compiler_options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_3 as u32,
    );
    compiler_options.set_source_language(shaderc::SourceLanguage::GLSL);
    compiler_options.set_optimization_level(if compile_info.optimize {
        shaderc::OptimizationLevel::Performance
    } else {
        shaderc::OptimizationLevel::Zero
    });
    if compile_info.debug_info {
        compiler_options.set_generate_debug_info();
    }

    compile_info
        .compile_defs
        .iter()
        .for_each(|(comp_def, comp_def_val)| {
            compiler_options.add_macro_definition(comp_def, *comp_def_val);
        });

    //let mut resolved_includes = std::collections::HashMap::<std::path::PathBuf, String>::new();

    compiler_options.set_include_callback(
        |requested_src: &str,
         _include_type: shaderc::IncludeType,
         requester: &str,
         _include_depth: usize|
         -> shaderc::IncludeCallbackResult {
            let requester_path = std::path::Path::new(requester)
                .canonicalize()
                .map_err(|e| e.to_string())?;
            let requested_path = std::path::Path::new(requested_src);

            let dir_requester = requester_path.parent().unwrap();
            let root_shader_dir = dir_requester
                .components()
                .take_while(|c| c.as_os_str() != std::ffi::OsStr::new("data"))
                .fold(std::path::PathBuf::new(), |p, c| p.join(c));

            //log::info!("requester {requester_path:?}, requested {requested_path:?}, dir requester {dir_requester:?}, root dir {root_shader_dir:?}");

            let resolved_path = if let Some(parent_path) = requested_path.parent() {
                if parent_path == std::path::Path::new("") {
                    dir_requester.join(requested_path)
                } else {
                    root_shader_dir.join("data/shaders").join(requested_path)
                }
            } else {
                dir_requester.join(requested_path)
            };

            log::info!("requested {requested_src} resolved to {resolved_path:?}");

            // if let Some(resolved_content) = resolved_includes.get(&resolved_path) {
            //     return Ok(shaderc::ResolvedInclude {
            //         resolved_name: resolved_path.to_str().unwrap().into(),
            //         content: resolved_content.clone(),
            //     });
            // }

            let include_content =
                std::fs::read_to_string(&resolved_path).map_err(|e| e.to_string())?;

            Ok(shaderc::ResolvedInclude {
                resolved_name: resolved_path.to_str().unwrap().into(),
                content: include_content,
            })
        },
    );

    let compiler = shaderc::Compiler::new()
        .ok_or_else(|| GraphicsError::ShadercGeneric("Failed to create compiler".into()))?;

    let preprocessed_shader = compiler.preprocess(
        shader_code,
        shader_tag,
        compile_info.entry_point.unwrap_or_else(|| "main"),
        Some(&compiler_options),
    )?;

    let compiled_bytecode = compiler.compile_into_spirv(
        &preprocessed_shader.as_text(),
        match shader_kind {
            ShaderKind::Vertex => shaderc::ShaderKind::Vertex,
            ShaderKind::Geometry => shaderc::ShaderKind::Geometry,
            ShaderKind::Fragment => shaderc::ShaderKind::Fragment,
            ShaderKind::Compute => shaderc::ShaderKind::Compute,
        },
        shader_tag,
        compile_info.entry_point.unwrap_or_else(|| "main"),
        Some(&compiler_options),
    )?;

    Ok(compiled_bytecode)
}

pub fn compile_shader<'a>(
    device: &ash::Device,
    compile_info: &'a ShaderCompileInfo,
) -> std::result::Result<(UniqueShaderModule, ash::vk::ShaderStageFlags), GraphicsError> {
    let (compiled_bytecode, shader_kind) = match compile_info.src {
        ShaderSource::File(ref p) => {
            let src_code = std::fs::read_to_string(p)
                .map_err(|e| GraphicsError::IoError { f: p.clone(), e })?;
            let shader_kind = p.try_into().map_err(|e| GraphicsError::ShadercGeneric(e))?;

            (
                compile_shader_impl(&src_code, shader_kind, p.to_str().unwrap(), compile_info)?,
                shader_kind,
            )
        }
        ShaderSource::String(src_code, shader_kind) => (
            compile_shader_impl(src_code, *shader_kind, "string shader", compile_info)?,
            *shader_kind,
        ),
    };

    let shader_module = UniqueShaderModule {
        module: unsafe {
            device.create_shader_module(
                &ash::vk::ShaderModuleCreateInfo::builder().code(compiled_bytecode.as_binary()),
                None,
            )
        }?,
        device: device as *const _,
    };

    Ok((shader_module, shader_kind.into()))
}

pub fn compile_and_reflect_shader<'a>(
    device: &ash::Device,
    compile_info: &'a ShaderCompileInfo,
) -> std::result::Result<
    (
        UniqueShaderModule,
        ash::vk::ShaderStageFlags,
        ShaderReflection,
    ),
    GraphicsError,
> {
    let (compiled_bytecode, shader_kind) = match compile_info.src {
        ShaderSource::File(ref p) => {
            let src_code = std::fs::read_to_string(p)
                .map_err(|e| GraphicsError::IoError { f: p.clone(), e })?;
            let shader_kind = p.try_into().map_err(|e| GraphicsError::ShadercGeneric(e))?;

            (
                compile_shader_impl(&src_code, shader_kind, p.to_str().unwrap(), compile_info)?,
                shader_kind,
            )
        }
        ShaderSource::String(src_code, shader_kind) => (
            compile_shader_impl(src_code, *shader_kind, "string shader", compile_info)?,
            *shader_kind,
        ),
    };

    let shader_module = UniqueShaderModule {
        module: unsafe {
            device.create_shader_module(
                &ash::vk::ShaderModuleCreateInfo::builder().code(compiled_bytecode.as_binary()),
                None,
            )
        }?,
        device: device as *const _,
    };

    let shader_reflection =
        reflect_shader_module(compiled_bytecode.as_binary_u8(), compile_info.entry_point)?;

    Ok((shader_module, shader_kind.into(), shader_reflection))
}

#[derive(Debug)]
pub struct ShaderReflection {
    pub inputs: Vec<ash::vk::VertexInputAttributeDescription>,
    pub inputs_stride: u32,
    pub descriptor_sets: std::collections::HashMap<u32, Vec<ash::vk::DescriptorSetLayoutBinding>>,
    pub push_constants: Vec<ash::vk::PushConstantRange>,
}

pub fn reflect_shader_module(
    spirv: &[u8],
    entry_point: Option<&str>,
) -> std::result::Result<ShaderReflection, GraphicsError> {
    let reflect = spirv_reflect::create_shader_module(spirv)
        .map_err(|e| GraphicsError::SpirVReflectionError(e))?;

    log::info!(
        "Generator {:?}, source lang {:?}, version {:?}, stage {:?}",
        reflect.get_generator(),
        reflect.get_source_language(),
        reflect.get_source_language_version(),
        reflect.get_shader_stage()
    );

    let set_descriptor_bindings = reflect
        .enumerate_descriptor_sets(entry_point)
        .map_err(|e| GraphicsError::SpirVReflectionError(e))?
        .iter()
        .map(|descriptor_set| {
            (
                descriptor_set.set,
                descriptor_set
                    .bindings
                    .iter()
                    .map(|descriptor_binding| {
                        *ash::vk::DescriptorSetLayoutBinding::builder()
                            .binding(descriptor_binding.binding)
                            .descriptor_type(spirv_reflect_descriptor_type_to_vk_descriptor_type(
                                descriptor_binding.descriptor_type,
                            ))
                            .descriptor_count(descriptor_binding.count)
                            .stage_flags(ash::vk::ShaderStageFlags::ALL)
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<std::collections::HashMap<u32, Vec<ash::vk::DescriptorSetLayoutBinding>>>();

    let (vertex_attribute_descriptions, stride) = {
        let mut bits_offset = 0u32;
        let reflected_attrs = reflect
            .enumerate_input_variables(entry_point)
            .map_err(|e| GraphicsError::SpirVReflectionError(e))?;
        dbg!(reflected_attrs);

        (
            reflect
                .enumerate_input_variables(entry_point)
                .map_err(|e| GraphicsError::SpirVReflectionError(e))?
                .iter()
                .filter(|input_var| {
                    !input_var.decoration_flags.intersects(
                        spirv_reflect::types::variable::ReflectDecorationFlags::BUILT_IN,
                    )
                })
                .map(|input_var| {
                    let attr_desc = *ash::vk::VertexInputAttributeDescription::builder()
                        .location(input_var.location)
                        .binding(0)
                        .format(spirv_reflect_format_to_vkformat(input_var.format))
                        .offset(bits_offset / 4);

                    bits_offset += input_var.numeric.scalar.width
                        * (if input_var.numeric.vector.component_count == 0 {
                            1
                        } else {
                            input_var.numeric.vector.component_count
                        });

                    attr_desc
                })
                .collect::<Vec<_>>(),
            bits_offset / 8,
        )
    };

    let push_constants = reflect
        .enumerate_push_constant_blocks(entry_point)
        .map_err(|e| GraphicsError::SpirVReflectionError(e))?
        .iter()
        .map(|push_const| {
            *ash::vk::PushConstantRange::builder()
                .stage_flags(ash::vk::ShaderStageFlags::ALL)
                .offset(push_const.offset)
                .size(push_const.size)
        })
        .collect::<Vec<_>>();

    Ok(ShaderReflection {
        inputs: vertex_attribute_descriptions,
        inputs_stride: stride,
        descriptor_sets: set_descriptor_bindings,
        push_constants,
    })
}

fn spirv_reflect_format_to_vkformat(fmt: spirv_reflect::types::ReflectFormat) -> ash::vk::Format {
    match fmt {
        spirv_reflect::types::ReflectFormat::Undefined => ash::vk::Format::UNDEFINED,
        spirv_reflect::types::ReflectFormat::R32_UINT => ash::vk::Format::R32_UINT,
        spirv_reflect::types::ReflectFormat::R32_SINT => ash::vk::Format::R32_SINT,
        spirv_reflect::types::ReflectFormat::R32_SFLOAT => ash::vk::Format::R32_SFLOAT,
        spirv_reflect::types::ReflectFormat::R32G32_UINT => ash::vk::Format::R32G32_UINT,
        spirv_reflect::types::ReflectFormat::R32G32_SINT => ash::vk::Format::R32G32_SINT,
        spirv_reflect::types::ReflectFormat::R32G32_SFLOAT => ash::vk::Format::R32G32_SFLOAT,
        spirv_reflect::types::ReflectFormat::R32G32B32_UINT => ash::vk::Format::R32G32B32_UINT,
        spirv_reflect::types::ReflectFormat::R32G32B32_SINT => ash::vk::Format::R32G32B32_SINT,
        spirv_reflect::types::ReflectFormat::R32G32B32_SFLOAT => ash::vk::Format::R32G32B32_SFLOAT,
        spirv_reflect::types::ReflectFormat::R32G32B32A32_UINT => {
            ash::vk::Format::R32G32B32A32_UINT
        }
        spirv_reflect::types::ReflectFormat::R32G32B32A32_SINT => {
            ash::vk::Format::R32G32B32A32_SINT
        }
        spirv_reflect::types::ReflectFormat::R32G32B32A32_SFLOAT => {
            ash::vk::Format::R32G32B32A32_SFLOAT
        }
    }
}

fn spirv_reflect_descriptor_type_to_vk_descriptor_type(
    dtype: spirv_reflect::types::descriptor::ReflectDescriptorType,
) -> ash::vk::DescriptorType {
    use ash::vk::DescriptorType;
    use spirv_reflect::types::descriptor::ReflectDescriptorType;

    match dtype {
        ReflectDescriptorType::Sampler => DescriptorType::SAMPLER,
        ReflectDescriptorType::CombinedImageSampler => DescriptorType::COMBINED_IMAGE_SAMPLER,
        ReflectDescriptorType::UniformBuffer => DescriptorType::UNIFORM_BUFFER,
        ReflectDescriptorType::StorageBuffer => DescriptorType::STORAGE_BUFFER,
        _ => panic!("Descriptor type {dtype:?} not supported"),
    }
}
