#![allow(dead_code)]

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum ShaderKind {
    Vertex,
    Geometry,
    Fragment,
    Compute,
}

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
    pub src: ShaderSource<'a>,
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
         include_type: shaderc::IncludeType,
         requester: &str,
         include_depth: usize|
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
    compile_info: &'a ShaderCompileInfo,
) -> std::result::Result<ash::vk::ShaderModule, GraphicsError> {
    let compiled_bytecode = match compile_info.src {
        ShaderSource::File(ref p) => {
            let src_code = std::fs::read_to_string(p)
                .map_err(|e| GraphicsError::IoError { f: p.clone(), e })?;
            let shader_kind = p.try_into().map_err(|e| GraphicsError::ShadercGeneric(e))?;

            compile_shader_impl(&src_code, shader_kind, p.to_str().unwrap(), compile_info)
        }
        ShaderSource::String(src_code, shader_kind) => {
            compile_shader_impl(src_code, shader_kind, "string shader", compile_info)
        }
    }?;

    let reflected_shader = reflect_shader_module(
        compiled_bytecode.as_binary_u8(),
        compile_info.entry_point.or_else(|| Some("main")),
    )?;

    Err(GraphicsError::ShadercGeneric("xxx".into()))
}

pub struct ReflectedShader {}

pub fn reflect_shader_module(
    spirv: &[u8],
    entry_point: Option<&str>,
) -> std::result::Result<ReflectedShader, GraphicsError> {
    let reflect = spirv_reflect::create_shader_module(spirv)
        .map_err(|e| GraphicsError::SpirVReflectionError(e))?;

    log::info!(
        "Generator {:?}, source lang {:?}, version {:?}",
        reflect.get_generator(),
        reflect.get_source_language(),
        reflect.get_source_language_version(),
    );

    let descriptor_bindings = reflect
        .enumerate_descriptor_bindings(entry_point)
        .map_err(|e| GraphicsError::SpirVReflectionError(e))?;

    descriptor_bindings
        .iter()
        .for_each(|db| log::info!("{db:?}"));

    let input_vars = reflect
        .enumerate_input_variables(entry_point)
        .map_err(|e| GraphicsError::SpirVReflectionError(e))?;

    let (vertex_attribute_descriptions, stride) = {
        let mut bits_offset = 0u32;

        (
            input_vars
                .iter()
                .filter(|input_var| {
                    input_var.storage_class
                        == spirv_reflect::types::variable::ReflectStorageClass::Input
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

    log::info!("input attributes {stride}");
    vertex_attribute_descriptions.iter().for_each(|input_var| {
        log::info!("{input_var:?}");
    });

    Ok(ReflectedShader {})
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
