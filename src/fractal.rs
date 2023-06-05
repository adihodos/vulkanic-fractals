use ash::vk::{
    BufferUsageFlags, ColorComponentFlags, CullModeFlags, DescriptorBufferInfo, DescriptorSet,
    DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding,
    DescriptorSetLayoutCreateInfo, DescriptorType, DynamicState, Extent2D, Format, FrontFace,
    GraphicsPipelineCreateInfo, MemoryPropertyFlags, Offset2D, Pipeline, PipelineBindPoint,
    PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateInfo,
    PipelineDynamicStateCreateInfo, PipelineInputAssemblyStateCreateInfo, PipelineLayout,
    PipelineLayoutCreateInfo, PipelineMultisampleStateCreateInfo,
    PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateInfo,
    PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo, PolygonMode,
    PrimitiveTopology, Rect2D, RenderPass, SampleCountFlags, ShaderStageFlags,
    VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, Viewport,
    WriteDescriptorSet,
};
use enum_iterator::{next_cycle, previous_cycle};

use crate::{
    vulkan_renderer::{compile_shader_from_file, UniqueBuffer, UniqueBufferMapping},
    FrameRenderContext, InputState, VulkanDeviceState, VulkanState,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq, enum_iterator::Sequence)]
#[repr(u32)]
pub enum Coloring {
    BlackWhite,
    Smooth,
    Log,
    Hsv,
    Rainbow,
    Palette,
}

trait FractalCoreParams {
    const MIN_ITERATIONS: u32;
    const MAX_ITERATIONS: u32;
    const ZOOM_IN_FACTOR: f32;
    const ZOOM_OUT_FACTOR: f32;
    const FRACTAL_XMIN: f32;
    const FRACTAL_XMAX: f32;
    const FRACTAL_YMIN: f32;
    const FRACTAL_YMAX: f32;
    const ESC_RADIUS_MIN: u32;
    const ESC_RADIUS_MAX: u32;

    const FRACTAL_HALF_WIDTH: f32 = (Self::FRACTAL_XMAX - Self::FRACTAL_XMIN) * 0.5f32;
    const FRACTAL_HALF_HEIGHT: f32 = (Self::FRACTAL_YMAX - Self::FRACTAL_YMIN) * 0.5f32;

    const VERTEX_SHADER_MODULE: &'static str;
    const FRAGMENT_SHADER_MODULE: &'static str;
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct FractalCore<T: FractalCoreParams> {
    screen_width: u32,
    screen_height: u32,
    iterations: u32,
    zoom: f32,
    ox: f32,
    oy: f32,
    coloring: Coloring,
    fxmin: f32,
    fxmax: f32,
    fymin: f32,
    fymax: f32,
    escape_radius: u32,
    _t: std::marker::PhantomData<T>,
}

impl<T> std::default::Default for FractalCore<T>
where
    T: FractalCoreParams,
{
    fn default() -> Self {
        Self {
            screen_width: 1024,
            screen_height: 1024,
            iterations: 64,
            coloring: Coloring::BlackWhite,
            zoom: 1f32,
            ox: 0f32,
            oy: 0f32,
            fxmin: -T::FRACTAL_HALF_WIDTH,
            fxmax: T::FRACTAL_HALF_WIDTH,
            fymin: -T::FRACTAL_HALF_HEIGHT,
            fymax: T::FRACTAL_HALF_HEIGHT,
            escape_radius: 2,
            _t: std::marker::PhantomData::default(),
        }
    }
}

impl<T> FractalCore<T>
where
    T: FractalCoreParams,
{
    pub fn input_handler(&mut self, input_state: &InputState) {
        use winit::event::ElementState;
        use winit::event::MouseButton;
        use winit::event::VirtualKeyCode;
        use winit::event::WindowEvent;

        match *input_state.event {
            WindowEvent::Resized(new_size) => {
                self.screen_resized(new_size.width, new_size.height);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let zoom_in = match delta {
                    winit::event::MouseScrollDelta::LineDelta(.., y) => y > 0f32,
                    winit::event::MouseScrollDelta::PixelDelta(winit::dpi::PhysicalPosition {
                        y,
                        ..
                    }) => y > 0f64,
                };

                if !zoom_in {
                    self.zoom_out();
                } else {
                    self.zoom_in();
                }
            }

            WindowEvent::KeyboardInput {
                input:
                    winit::event::KeyboardInput {
                        virtual_keycode,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match virtual_keycode {
                Some(VirtualKeyCode::PageUp) => {
                    self.previous_color();
                }

                Some(VirtualKeyCode::PageDown) => {
                    self.next_color();
                }

                Some(VirtualKeyCode::NumpadSubtract) => {
                    self.decrease_iterations();
                }

                Some(VirtualKeyCode::NumpadAdd) => {
                    self.increase_iterations();
                }

                Some(VirtualKeyCode::Insert) => {
                    self.increase_escape_radius();
                }

                Some(VirtualKeyCode::Delete) => {
                    self.decrease_escape_radius();
                }
                _ => {}
            },

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => match button {
                MouseButton::Left => {
                    self.center_moved(
                        input_state.cursor_pos.0,
                        input_state.cursor_pos.1,
                        input_state.control_down,
                    );
                }
                MouseButton::Right => {
                    self.zoom_out();
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn center_moved(&mut self, cx: f32, cy: f32, zoom: bool) {
        let (cx, cy) = screen_coords_to_complex_coords(
            cx,
            cy,
            T::FRACTAL_XMIN,
            T::FRACTAL_XMAX,
            T::FRACTAL_YMIN,
            T::FRACTAL_YMAX,
            self.screen_width as f32,
            self.screen_height as f32,
        );

        if zoom {
            self.zoom *= T::ZOOM_IN_FACTOR;
        }

        self.ox += cx * self.zoom;
        self.oy += cy * self.zoom;

        self.fxmin = self.ox - T::FRACTAL_HALF_WIDTH * self.zoom;
        self.fxmax = self.ox + T::FRACTAL_HALF_WIDTH * self.zoom;
        self.fymin = self.oy - T::FRACTAL_HALF_HEIGHT * self.zoom;
        self.fymax = self.oy + T::FRACTAL_HALF_HEIGHT * self.zoom;
    }

    fn zoom_out(&mut self) {
        self.zoom = (self.zoom * T::ZOOM_OUT_FACTOR).min(1f32);
        self.fxmin = self.ox - T::FRACTAL_HALF_WIDTH * self.zoom;
        self.fxmax = self.ox + T::FRACTAL_HALF_WIDTH * self.zoom;
        self.fymin = self.oy - T::FRACTAL_HALF_HEIGHT * self.zoom;
        self.fymax = self.oy + T::FRACTAL_HALF_HEIGHT * self.zoom;
    }

    fn zoom_in(&mut self) {
        self.zoom = (self.zoom * T::ZOOM_IN_FACTOR).max(0f32);
        self.fxmin = self.ox - T::FRACTAL_HALF_WIDTH * self.zoom;
        self.fxmax = self.ox + T::FRACTAL_HALF_WIDTH * self.zoom;
        self.fymin = self.oy - T::FRACTAL_HALF_HEIGHT * self.zoom;
        self.fymax = self.oy + T::FRACTAL_HALF_HEIGHT * self.zoom;
    }

    fn screen_resized(&mut self, width: u32, height: u32) {
        self.screen_width = width;
        self.screen_height = height;
    }

    fn increase_iterations(&mut self) {
        self.iterations = (self.iterations * 2).min(T::MAX_ITERATIONS);
    }

    fn decrease_iterations(&mut self) {
        self.iterations = (self.iterations / 2).max(T::MIN_ITERATIONS);
    }

    fn next_color(&mut self) {
        self.coloring = next_cycle(&self.coloring).unwrap();
    }

    fn previous_color(&mut self) {
        self.coloring = previous_cycle(&self.coloring).unwrap();
    }

    fn increase_escape_radius(&mut self) {
        self.escape_radius = (self.escape_radius * 2).min(T::ESC_RADIUS_MAX);
    }

    fn decrease_escape_radius(&mut self) {
        self.escape_radius = (self.escape_radius / 2).max(T::ESC_RADIUS_MIN);
    }

    fn reset(&mut self) {
        *self = Self {
            screen_width: self.screen_width,
            screen_height: self.screen_height,
            ..Default::default()
        };
    }

    pub fn do_ui(&mut self, ui: &imgui::Ui) {
        //
        // coloring algorithm
        if let Some(_cb) = ui.begin_combo("Coloring algorithm", format!("{:?}", self.coloring)) {
            let mut selected = self.coloring;
            for item in enum_iterator::all::<Coloring>() {
                if selected == item {
                    ui.set_item_default_focus();
                }

                let clicked = ui
                    .selectable_config(format!("{:?}", item))
                    .selected(selected == item)
                    .build();

                // When item is clicked, store it
                if clicked {
                    selected = item;
                    self.coloring = item;
                }
            }
        }

        let mut escape_radius = self.escape_radius;
        ui.slider_config("Escape radius", T::ESC_RADIUS_MIN, T::ESC_RADIUS_MAX)
            .build(&mut escape_radius);
        self.escape_radius = escape_radius;

        let mut iterations = self.iterations;

        ui.slider_config("Max iterations", T::MIN_ITERATIONS, T::MAX_ITERATIONS)
            .build(&mut iterations);

        self.iterations = iterations;

        ui.separator();
        ui.label_text("", "Info");

        ui.text_colored(
            [1f32, 0f32, 0f32, 1f32],
            format!("Screen {}x{}", self.screen_width, self.screen_height),
        );

        let cursor_pos = ui.io().mouse_pos;
        ui.text_colored(
            [1f32, 0f32, 0f32, 1f32],
            format!("Cursor position ({}, {})", cursor_pos[0], cursor_pos[1]),
        );

        ui.text_colored(
            [1f32, 0f32, 0f32, 1f32],
            format!("Center: ({}, {})", self.ox, self.oy),
        );
        ui.text_colored(
            [1f32, 0f32, 0f32, 1f32],
            format!(
                "Domain: ({}, {}) x ({}, {})",
                self.fxmin, self.fymin, self.fxmax, self.fymax
            ),
        );
        ui.text_colored(
            [1f32, 0f32, 0f32, 1f32],
            format!("Zoom: {}", 1f32 / self.zoom),
        );
    }
}

#[derive(Copy, Clone, Debug)]
struct MandelbrotParams {}

impl FractalCoreParams for MandelbrotParams {
    const MIN_ITERATIONS: u32 = 4;
    const MAX_ITERATIONS: u32 = 2048;
    const ZOOM_IN_FACTOR: f32 = 0.85f32;
    const ZOOM_OUT_FACTOR: f32 = 2f32;
    const FRACTAL_XMIN: f32 = -2f32;
    const FRACTAL_XMAX: f32 = 2f32;
    const FRACTAL_YMIN: f32 = -1f32;
    const FRACTAL_YMAX: f32 = 1f32;
    const ESC_RADIUS_MIN: u32 = 2;
    const ESC_RADIUS_MAX: u32 = 4096;

    const VERTEX_SHADER_MODULE: &'static str = "data/shaders/fractal.vert";
    const FRAGMENT_SHADER_MODULE: &'static str = "data/shaders/mandelbrot.frag";
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, enum_iterator::Sequence)]
#[repr(u32)]
enum JuliaIterationType {
    Quadratic,
    Sine,
    Cosine,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct JuliaCPU2GPU {
    core: FractalCore<JuliaCPU2GPU>,
    c_x: f32,
    c_y: f32,
    iteration: JuliaIterationType,
}

impl FractalCoreParams for JuliaCPU2GPU {
    const MIN_ITERATIONS: u32 = 4;
    const MAX_ITERATIONS: u32 = 2048;
    const ZOOM_IN_FACTOR: f32 = 0.85f32;
    const ZOOM_OUT_FACTOR: f32 = 2f32;
    const FRACTAL_XMIN: f32 = -2f32;
    const FRACTAL_XMAX: f32 = 2f32;
    const FRACTAL_YMIN: f32 = -1f32;
    const FRACTAL_YMAX: f32 = 1f32;
    const ESC_RADIUS_MIN: u32 = 2;
    const ESC_RADIUS_MAX: u32 = 4096;
    const VERTEX_SHADER_MODULE: &'static str = "data/shaders/fractal.vert";
    const FRAGMENT_SHADER_MODULE: &'static str = "data/shaders/julia.frag";
}

impl std::default::Default for JuliaCPU2GPU {
    fn default() -> Self {
        Self {
            c_x: -0.7f32,
            c_y: -0.3f32,
            iteration: JuliaIterationType::Quadratic,
            core: Default::default(),
        }
    }
}

pub struct Julia {
    params: JuliaCPU2GPU,
    gpu_state: FractalGPUState<JuliaCPU2GPU>,
}

impl Julia {
    pub fn new(vks: &VulkanState) -> Julia {
        Self {
            params: JuliaCPU2GPU::default(),
            gpu_state: FractalGPUState::new(vks),
        }
    }

    pub fn input_handler(&mut self, input_state: &InputState) {
        use winit::event::WindowEvent;

        match *input_state.event {
            WindowEvent::KeyboardInput {
                input:
                    winit::event::KeyboardInput {
                        virtual_keycode: Some(winit::event::VirtualKeyCode::Back),
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                self.params.c_x = -0.7f32;
                self.params.c_y = -0.3f32;
                self.params.core.reset();
            }

            _ => self.params.core.input_handler(input_state),
        }
    }

    pub fn render(&mut self, vks: &VulkanState, context: &FrameRenderContext) {
        self.gpu_state.render(vks, context, &[self.params]);
    }

    pub fn do_ui(&mut self, ui: &mut imgui::Ui) {
        ui.window("Julia fractal parameters")
            .always_auto_resize(true)
            .build(|| {
                self.params.core.do_ui(ui);

                let mut c_ptr: [f32; 2] = [self.params.c_x, self.params.c_y];

                if ui.input_float2("C point", &mut c_ptr).build() {
                    self.params.c_x = c_ptr[0];
                    self.params.c_y = c_ptr[1];
                }

                ui.new_line();
                ui.text("Iteration:");

                enum_iterator::all::<JuliaIterationType>().for_each(|it| {
                    if ui.radio_button_bool(format!("{:?}", it), self.params.iteration == it) {
                        log::info!("Clicked button {:?}", it,);
                        self.params.iteration = it;
                    }
                });
            });
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct MandelbrotCPU2GPU {
    core: FractalCore<MandelbrotParams>,
}

impl FractalCoreParams for MandelbrotCPU2GPU {
    const MIN_ITERATIONS: u32 = 4;
    const MAX_ITERATIONS: u32 = 2048;
    const ZOOM_IN_FACTOR: f32 = 0.85f32;
    const ZOOM_OUT_FACTOR: f32 = 2f32;
    const FRACTAL_XMIN: f32 = -2f32;
    const FRACTAL_XMAX: f32 = 2f32;
    const FRACTAL_YMIN: f32 = -1f32;
    const FRACTAL_YMAX: f32 = 1f32;
    const ESC_RADIUS_MIN: u32 = 2;
    const ESC_RADIUS_MAX: u32 = 4096;
    const VERTEX_SHADER_MODULE: &'static str = "data/shaders/fractal.vert";
    const FRAGMENT_SHADER_MODULE: &'static str = "data/shaders/mandelbrot.frag";
}

pub struct Mandelbrot {
    params: MandelbrotCPU2GPU,
    gpu_state: FractalGPUState<MandelbrotCPU2GPU>,
}

impl Mandelbrot {
    pub fn new(vks: &VulkanState) -> Mandelbrot {
        Self {
            params: MandelbrotCPU2GPU {
                core: FractalCore::default(),
            },
            gpu_state: FractalGPUState::new(vks),
        }
    }

    pub fn input_handler(&mut self, input_state: &InputState) {
        use winit::event::WindowEvent;

        match *input_state.event {
            WindowEvent::KeyboardInput {
                input:
                    winit::event::KeyboardInput {
                        virtual_keycode: Some(winit::event::VirtualKeyCode::Back),
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                self.params.core.reset();
            }

            _ => self.params.core.input_handler(input_state),
        }
    }

    pub fn render(&mut self, vks: &VulkanState, context: &FrameRenderContext) {
        self.gpu_state.render(vks, context, &[self.params]);
    }

    pub fn do_ui(&mut self, ui: &mut imgui::Ui) {
        ui.window("Mandelbrot fractal parameters")
            .always_auto_resize(true)
            .build(|| {
                self.params.core.do_ui(ui);
            });
    }
}

struct FractalGPUState<T: FractalCoreParams + Copy> {
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    _descriptor_set_layout: DescriptorSetLayout,
    buffer: UniqueBuffer,
    ubo_params: UniqueBuffer,
    descriptor_ubo_buffer: DescriptorSet,
    _marker: std::marker::PhantomData<T>,
}

impl<T> FractalGPUState<T>
where
    T: FractalCoreParams + Copy,
{
    fn new(vks: &VulkanState) -> Self {
        let (pipeline, pipeline_layout, descriptor_set_layout) =
            Self::create_graphics_pipeline(&vks.ds, vks.renderpass);

        let ubo_params = UniqueBuffer::new::<T>(
            vks,
            BufferUsageFlags::UNIFORM_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            vks.swapchain.max_frames as usize,
        );

        let descriptor_ubo_buffer = unsafe {
            vks.ds.device.allocate_descriptor_sets(
                &DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(vks.ds.descriptor_pool)
                    .set_layouts(&[descriptor_set_layout]),
            )
        }
        .expect("Failed to allocate descriptor set")[0];

        unsafe {
            vks.ds.device.update_descriptor_sets(
                &[*WriteDescriptorSet::builder()
                    .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .dst_array_element(0)
                    .dst_binding(0)
                    .dst_set(descriptor_ubo_buffer)
                    .buffer_info(&[*DescriptorBufferInfo::builder()
                        .buffer(ubo_params.handle)
                        .offset(0)
                        .range(ubo_params.item_aligned_size as u64)])],
                &[],
            );
        }

        Self {
            pipeline,
            pipeline_layout,
            _descriptor_set_layout: descriptor_set_layout,
            buffer: UniqueBuffer::new::<f32>(
                vks,
                BufferUsageFlags::VERTEX_BUFFER,
                MemoryPropertyFlags::DEVICE_LOCAL | MemoryPropertyFlags::HOST_VISIBLE,
                64,
            ),
            ubo_params,
            descriptor_ubo_buffer,
            _marker: std::marker::PhantomData::<T>,
        }
    }

    fn render(&mut self, vks: &VulkanState, context: &FrameRenderContext, gpu_data: &[T]) {
        let render_area = Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent: context.fb_size,
        };

        //
        // copy params
        UniqueBufferMapping::new(
            &self.ubo_params,
            &vks.ds,
            Some(self.ubo_params.item_aligned_size * context.current_frame_id as usize),
            Some(self.ubo_params.item_aligned_size),
        )
        .write_data(gpu_data);

        unsafe {
            vks.ds.device.cmd_set_viewport(
                context.cmd_buff,
                0,
                &[Viewport {
                    x: 0f32,
                    y: 0f32,
                    width: context.fb_size.width as f32,
                    height: context.fb_size.height as f32,
                    min_depth: 0f32,
                    max_depth: 1f32,
                }],
            );

            vks.ds
                .device
                .cmd_set_scissor(context.cmd_buff, 0, &[render_area]);

            vks.ds.device.cmd_bind_pipeline(
                context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            vks.ds.device.cmd_bind_vertex_buffers(
                context.cmd_buff,
                0,
                &[self.buffer.handle],
                &[0u64],
            );

            vks.ds.device.cmd_bind_descriptor_sets(
                context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_ubo_buffer],
                &[self.ubo_params.item_aligned_size as u32 * context.current_frame_id as u32],
            );

            vks.ds.device.cmd_draw(context.cmd_buff, 6, 1, 0, 0);
        }
    }

    fn create_graphics_pipeline(
        ds: &VulkanDeviceState,
        renderpass: RenderPass,
    ) -> (Pipeline, PipelineLayout, DescriptorSetLayout) {
        let descriptor_set_layout = unsafe {
            ds.device.create_descriptor_set_layout(
                &DescriptorSetLayoutCreateInfo::builder().bindings(&[
                    DescriptorSetLayoutBinding::builder()
                        .binding(0)
                        .descriptor_count(1)
                        .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .stage_flags(ShaderStageFlags::FRAGMENT)
                        .build(),
                ]),
                None,
            )
        }
        .expect("Failed to create descriptor set layout");

        let pipeline_layout = unsafe {
            ds.device.create_pipeline_layout(
                &PipelineLayoutCreateInfo::builder().set_layouts(&[descriptor_set_layout]),
                None,
            )
        }
        .expect("Failed to create pipeline layout");

        let vsm = compile_shader_from_file(T::VERTEX_SHADER_MODULE, &ds.device).unwrap();
        let fsm = compile_shader_from_file(T::FRAGMENT_SHADER_MODULE, &ds.device).unwrap();

        use std::ffi::CStr;
        let pipeline = unsafe {
            ds.device.create_graphics_pipelines(
                ash::vk::PipelineCache::null(),
                &[*GraphicsPipelineCreateInfo::builder()
                    .stages(&[
                        *PipelineShaderStageCreateInfo::builder()
                            .module(*vsm)
                            .stage(ShaderStageFlags::VERTEX)
                            .name(CStr::from_bytes_with_nul(b"main\0" as &[u8]).unwrap()),
                        *PipelineShaderStageCreateInfo::builder()
                            .module(*fsm)
                            .stage(ShaderStageFlags::FRAGMENT)
                            .name(CStr::from_bytes_with_nul(b"main\0" as &[u8]).unwrap()),
                    ])
                    .vertex_input_state(
                        &PipelineVertexInputStateCreateInfo::builder()
                            .vertex_attribute_descriptions(&[
                                *VertexInputAttributeDescription::builder()
                                    .binding(0)
                                    .format(Format::R32G32_SFLOAT)
                                    .location(0)
                                    .offset(0),
                                *VertexInputAttributeDescription::builder()
                                    .binding(0)
                                    .format(Format::R32G32B32A32_SFLOAT)
                                    .location(1)
                                    .offset(8),
                            ])
                            .vertex_binding_descriptions(&[
                                *VertexInputBindingDescription::builder()
                                    .binding(0)
                                    .input_rate(VertexInputRate::VERTEX)
                                    .stride(24),
                            ]),
                    )
                    .input_assembly_state(
                        &PipelineInputAssemblyStateCreateInfo::builder()
                            .topology(PrimitiveTopology::TRIANGLE_LIST),
                    )
                    .rasterization_state(
                        &PipelineRasterizationStateCreateInfo::builder()
                            .polygon_mode(PolygonMode::FILL)
                            .cull_mode(CullModeFlags::BACK)
                            .front_face(FrontFace::COUNTER_CLOCKWISE)
                            .line_width(1f32)
                            .depth_clamp_enable(false),
                    )
                    .multisample_state(
                        &PipelineMultisampleStateCreateInfo::builder()
                            .rasterization_samples(SampleCountFlags::TYPE_1),
                    )
                    .color_blend_state(
                        &PipelineColorBlendStateCreateInfo::builder()
                            .attachments(&[*PipelineColorBlendAttachmentState::builder()
                                .color_write_mask(ColorComponentFlags::RGBA)]),
                    )
                    .dynamic_state(
                        &PipelineDynamicStateCreateInfo::builder()
                            .dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR]),
                    )
                    .viewport_state(
                        &PipelineViewportStateCreateInfo::builder()
                            .viewports(&[*Viewport::builder()
                                .x(0f32)
                                .y(0f32)
                                .width(ds.surface.image_size.width as f32)
                                .height(ds.surface.image_size.height as f32)
                                .min_depth(0f32)
                                .max_depth(1f32)])
                            .scissors(&[Rect2D {
                                offset: Offset2D { x: 0, y: 0 },
                                extent: Extent2D {
                                    width: ds.surface.image_size.width,
                                    height: ds.surface.image_size.height,
                                },
                            }]),
                    )
                    .layout(pipeline_layout)
                    .render_pass(renderpass)
                    .subpass(0)],
                None,
            )
        }
        .expect("Failed to create graphics pipeline ... ");

        (pipeline[0], pipeline_layout, descriptor_set_layout)
    }
}

fn screen_coords_to_complex_coords(
    px: f32,
    py: f32,
    dxmin: f32,
    dxmax: f32,
    dymin: f32,
    dymax: f32,
    screen_width: f32,
    screen_height: f32,
) -> (f32, f32) {
    let x = (px / screen_width) * (dxmax - dxmin) + dxmin;
    let y = (py / screen_height) * (dymax - dymin) + dymin;

    (x, y)
}
