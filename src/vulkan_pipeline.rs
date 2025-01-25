use ash::vk::{
    BlendFactor, BlendOp, ColorComponentFlags, CompareOp, CullModeFlags, DescriptorBindingFlags,
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutBindingFlagsCreateInfo,
    DescriptorSetLayoutCreateInfo, DynamicState, FrontFace, GraphicsPipelineCreateInfo, Pipeline,
    PipelineColorBlendStateCreateInfo, PipelineDepthStencilStateCreateInfo,
    PipelineDynamicStateCreateInfo, PipelineLayout, PipelineMultisampleStateCreateInfo,
    PipelineRasterizationStateCreateInfo, PipelineRenderingCreateInfo,
    PipelineShaderStageCreateInfo, PipelineVertexInputStateCreateInfo, PolygonMode,
    PushConstantRange, SampleCountFlags, ShaderStageFlags, VertexInputAttributeDescription,
    VertexInputRate,
};

use crate::{
    vulkan_renderer::{GraphicsError, RenderState, VulkanRenderer},
    vulkan_shader::{DescriptorSetLayoutBindingInfo, ShaderSource},
};

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
    device: *const ash::Device,
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

pub struct InputAssemblyState {
    pub stride: u32,
    pub input_rate: ash::vk::VertexInputRate,
    pub vertex_descriptions: Vec<ash::vk::VertexInputAttributeDescription>,
}

#[derive(Default)]
pub struct GraphicsPipelineSetupHelper<'a> {
    shader_stages: Vec<ShaderSource<'a>>,
    input_assembly_state: Option<InputAssemblyState>,
    rasterization_state: Option<PipelineRasterizationStateCreateInfo<'a>>,
    multisample_state: Option<PipelineMultisampleStateCreateInfo<'a>>,
    depth_stencil_state: Option<PipelineDepthStencilStateCreateInfo<'a>>,
    color_blend_state: Option<PipelineColorBlendStateCreateInfo<'a>>,
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

    pub fn set_raster_state(mut self, raster: PipelineRasterizationStateCreateInfo<'a>) -> Self {
        self.rasterization_state = Some(raster);
        self
    }

    pub fn set_multisample_state(mut self, ms: PipelineMultisampleStateCreateInfo<'a>) -> Self {
        self.multisample_state = Some(ms);
        self
    }

    pub fn set_depth_stencil_state(mut self, ds: PipelineDepthStencilStateCreateInfo<'a>) -> Self {
        self.depth_stencil_state = Some(ds);
        self
    }

    pub fn set_colorblend_state(mut self, cb: PipelineColorBlendStateCreateInfo<'a>) -> Self {
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
            use crate::vulkan_shader::*;

            let mut module_data: Vec<(UniqueShaderModule, ShaderReflection)> = vec![];
            let mut shader_stage_create_info: Vec<PipelineShaderStageCreateInfo> = vec![];

            for shader_src in self.shader_stages.into_iter() {
                let (shader_module, shader_stage, shader_reflection) = compile_and_reflect_shader(
                    renderer.logical(),
                    &crate::vulkan_shader::ShaderCompileInfo {
                        src: &shader_src,
                        entry_point: Some("main"),
                        compile_defs: &[],
                        optimize: false,
                        debug_info: true,
                    },
                )?;

                dbg!(&shader_reflection);

                shader_stage_create_info.push(
                    PipelineShaderStageCreateInfo::default()
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
                std::collections::HashMap::<u32, DescriptorSetLayoutBindingInfo>::new();

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
                        .or_insert( DescriptorSetLayoutBindingInfo {
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

            let descriptor_set_layout : Vec<ash::vk::DescriptorSetLayout> = Result::from_iter((0..=max_set_id)
                .map(|set_id| {
                    if let Some(set_entry) = descriptor_set_table.get(&set_id) {
                        let set_entry = [DescriptorSetLayoutBinding::default()
                            .binding(set_entry.binding)
                            .descriptor_type(set_entry.descriptor_type)
                            .descriptor_count(set_entry.descriptor_count)
                            .stage_flags(set_entry.stage_flags)];

                        let binding_flags = [DescriptorBindingFlags::PARTIALLY_BOUND | DescriptorBindingFlags::UPDATE_AFTER_BIND | DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT];
                        let mut set_layout_binding_flags = DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

                        unsafe {
                            renderer
                                .logical()
                                .create_descriptor_set_layout(
                                    &DescriptorSetLayoutCreateInfo::default()
                                        .push_next(&mut set_layout_binding_flags)
                                        .flags(ash::vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                                        .bindings(&set_entry),
                                    None,
                                )
                        }
                    } else {
                        Ok(renderer.empty_descriptor_set_layout())
                    }
                })
                .collect::<Vec<_>>())?;

            PipelineData::Owned {
                layout: unsafe {
                    renderer.logical().create_pipeline_layout(
                        &ash::vk::PipelineLayoutCreateInfo::default()
                            .set_layouts(&descriptor_set_layout)
                            .push_constant_ranges(&push_constant_ranges),
                        None,
                    )
                }?,
                descriptor_set_layout,
            }
        };

        let input_assembly_state = ash::vk::PipelineInputAssemblyStateCreateInfo::default()
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

        let std_vertex_binding = [ash::vk::VertexInputBindingDescription::default()
            .stride(inputs_stride)
            .input_rate(input_rate)
            .binding(0)];

        let vertex_input_state = self
            .input_assembly_state
            .as_ref()
            .map(|ia| {
                PipelineVertexInputStateCreateInfo::default()
                    .vertex_attribute_descriptions(&ia.vertex_descriptions)
                    .vertex_binding_descriptions(&std_vertex_binding)
            })
            .unwrap_or_else(|| {
                reflected_vertex_inputs_attribute_descriptions
                    .as_ref()
                    .map_or_else(
                        || PipelineVertexInputStateCreateInfo::default(),
                        |(_, vertex_inputs)| {
                            PipelineVertexInputStateCreateInfo::default()
                                .vertex_attribute_descriptions(vertex_inputs)
                                .vertex_binding_descriptions(&std_vertex_binding)
                        },
                    )
            });

        let rasterization_state = self.rasterization_state.unwrap_or_else(|| {
            PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(PolygonMode::FILL)
                .cull_mode(CullModeFlags::BACK)
                .front_face(FrontFace::COUNTER_CLOCKWISE)
                .depth_bias_enable(false)
                .line_width(1f32)
        });

        let multisample_state = self.multisample_state.unwrap_or_else(|| {
            PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(SampleCountFlags::TYPE_1)
        });

        let depth_stencil_state = self.depth_stencil_state.unwrap_or_else(|| {
            PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(CompareOp::LESS)
                .depth_bounds_test_enable(true)
                .min_depth_bounds(0f32)
                .max_depth_bounds(1f32)
        });

        let colorblend_attachments = [ash::vk::PipelineColorBlendAttachmentState::default()
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
            PipelineColorBlendStateCreateInfo::default()
                .blend_constants([1f32; 4])
                .attachments(&colorblend_attachments)
        });

        let dynamic_state =
            PipelineDynamicStateCreateInfo::default().dynamic_states(&self.dynamic_state);

        let viewport_state = ash::vk::PipelineViewportStateCreateInfo::default();

        // let graphics_pipeline_create_info = {
        //     let mut pipeline_create_info = GraphicsPipelineCreateInfo::default()
        //         .input_assembly_state(&input_assembly_state)
        //         .stages(&pipeline_shader_stage_create_info)
        //         .vertex_input_state(&vertex_input_state)
        //         .rasterization_state(&rasterization_state)
        //         .multisample_state(&multisample_state)
        //         .depth_stencil_state(&depth_stencil_state)
        //         .color_blend_state(&colorblend_state)
        //         .viewport_state(&viewport_state)
        //         .dynamic_state(&dynamic_state)
        //         .layout(pipeline_data.layout());

        let pipeline_handles = match renderer.render_state() {
            RenderState::Renderpass(pass) => unsafe {
                renderer.logical().create_graphics_pipelines(
                    ash::vk::PipelineCache::null(),
                    &[GraphicsPipelineCreateInfo::default()
                        .input_assembly_state(&input_assembly_state)
                        .stages(&pipeline_shader_stage_create_info)
                        .vertex_input_state(&vertex_input_state)
                        .rasterization_state(&rasterization_state)
                        .multisample_state(&multisample_state)
                        .depth_stencil_state(&depth_stencil_state)
                        .color_blend_state(&colorblend_state)
                        .viewport_state(&viewport_state)
                        .dynamic_state(&dynamic_state)
                        .layout(pipeline_data.layout())
                        .render_pass(pass)
                        .subpass(0)],
                    None,
                )
            },

            RenderState::Dynamic {
                color_attachments,
                depth_attachments,
                stencil_attachments,
            } => {
                let mut render_create_info = PipelineRenderingCreateInfo::default()
                    .color_attachment_formats(&color_attachments)
                    .depth_attachment_format(depth_attachments[0])
                    .stencil_attachment_format(stencil_attachments[0]);

                unsafe {
                    renderer.logical().create_graphics_pipelines(
                        ash::vk::PipelineCache::null(),
                        &[GraphicsPipelineCreateInfo::default()
                            .push_next(&mut render_create_info)
                            .input_assembly_state(&input_assembly_state)
                            .stages(&pipeline_shader_stage_create_info)
                            .vertex_input_state(&vertex_input_state)
                            .rasterization_state(&rasterization_state)
                            .multisample_state(&multisample_state)
                            .depth_stencil_state(&depth_stencil_state)
                            .color_blend_state(&colorblend_state)
                            .viewport_state(&viewport_state)
                            .dynamic_state(&dynamic_state)
                            .layout(pipeline_data.layout())],
                        None,
                    )
                }
            }
        }
        .map_err(|(_, e)| e)?;

        Ok(UniquePipeline {
            device: renderer.logical_raw(),
            handle: pipeline_handles[0],
            data: pipeline_data,
        })
    }
}
