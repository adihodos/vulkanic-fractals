use std::mem::size_of;

use ash::vk::{
    BorderColor, BufferUsageFlags, ColorComponentFlags, ComponentSwizzle, CullModeFlags,
    DynamicState, Extent2D, Extent3D, Filter, Format, FrontFace, GraphicsPipelineCreateInfo,
    ImageLayout, ImageTiling, ImageType, ImageUsageFlags, ImageViewCreateInfo, MemoryPropertyFlags,
    Offset2D, Pipeline, PipelineBindPoint, PipelineColorBlendAttachmentState,
    PipelineColorBlendStateCreateInfo, PipelineDynamicStateCreateInfo,
    PipelineInputAssemblyStateCreateInfo, PipelineLayout, PipelineMultisampleStateCreateInfo,
    PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateInfo,
    PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo, PolygonMode,
    PrimitiveTopology, Rect2D, RenderPass, SampleCountFlags, SamplerAddressMode, SamplerCreateInfo,
    SamplerMipmapMode, ShaderStageFlags, SharingMode, Viewport,
};
use enum_iterator::{next_cycle, previous_cycle};
use palette::chromatic_adaptation::AdaptInto;

use crate::{
    vulkan_renderer::{
        compile_shader_from_file, BindlessResourceHandle, BindlessResourceSystem,
        FrameRenderContext, UniqueBuffer, UniqueBufferMapping, UniqueImage, UniqueImageView,
        UniqueSampler, VulkanDeviceState, VulkanState,
    },
    InputState,
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
    const BINDLESS_FRAGMENT_SHADER_MODULE: &'static str;
}

#[derive(Copy, Clone, Debug)]
#[repr(C, align(16))]
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
    palette: u32,
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
            palette: 0,
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
            format!(
                "Screen dimensions: {}x{}",
                self.screen_width, self.screen_height
            ),
        );

        let cursor_pos = ui.io().mouse_pos;
        ui.text_colored(
            [1f32, 0f32, 0f32, 1f32],
            format!("Cursor position: ({}, {})", cursor_pos[0], cursor_pos[1]),
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
            format!("Zoom: {}x", 1f32 / self.zoom),
        );
    }
}

#[derive(Copy, Clone, Debug)]
struct MandelbrotParams {}

#[derive(Copy, Clone, Debug, Eq, PartialEq, enum_iterator::Sequence)]
#[repr(u32)]
enum JuliaIterationType {
    Quadratic,
    Sine,
    Cosine,
    Cubic,
}

#[derive(Copy, Clone, Debug)]
#[repr(C, align(16))]
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
    const FRACTAL_YMIN: f32 = -1.5f32;
    const FRACTAL_YMAX: f32 = 1.5f32;
    const ESC_RADIUS_MIN: u32 = 2;
    const ESC_RADIUS_MAX: u32 = 4096;
    const VERTEX_SHADER_MODULE: &'static str = "data/shaders/fractal.vert";
    const FRAGMENT_SHADER_MODULE: &'static str = "data/shaders/julia.frag";
    const BINDLESS_FRAGMENT_SHADER_MODULE: &'static str = "data/shaders/julia.bindless.frag";
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

impl JuliaCPU2GPU {
    fn new(c_x: f32, c_y: f32, iteration: JuliaIterationType) -> JuliaCPU2GPU {
        Self {
            core: Default::default(),
            c_x,
            c_y,
            iteration,
        }
    }
}

pub struct Julia {
    params: JuliaCPU2GPU,
    gpu_state: FractalGPUState<JuliaCPU2GPU>,
    point_quadratic_idx: usize,
    point_sine_idx: usize,
    point_cosine_idx: usize,
    point_cubic_idx: usize,
}

struct InterestingPoint {
    coords: [f32; 2],
    desc: &'static str,
}

impl Julia {
    const INTERESTING_POINTS_QUADRATIC: [InterestingPoint; 8] = [
        InterestingPoint {
            coords: [0f32, 1f32],
            desc: "(0.0 + 1.0 * i) - dentrite fractal",
        },
        InterestingPoint {
            coords: [-0.123f32, 0.745f32],
            desc: "(-0.123f32 + 0.745 * i) - douady's rabbit fractal",
        },
        InterestingPoint {
            coords: [-0.750f32, 0f32],
            desc: "(-0.750 + 0 * i) - san marco fractal",
        },
        InterestingPoint {
            coords: [-0.391f32, -0.587f32],
            desc: "(-0.391 + -0.587f32 * i) - siegel disk fractal",
        },
        InterestingPoint {
            coords: [-0.7f32, -0.3f32],
            desc: "(-0.7f32 + -0.3f32 * i) - NEAT cauliflower thingy",
        },
        InterestingPoint {
            coords: [-0.75f32, -0.2f32],
            desc: "(-0.75f32 + -0.2f32 * i) - galaxies",
        },
        InterestingPoint {
            coords: [-0.75f32, 0.15f32],
            desc: "(-0.75f32 + 0.15f32 * i) - groovy",
        },
        InterestingPoint {
            coords: [-0.7f32, 0.35f32],
            desc: "(-0.7f32 + 0.35f32 * i) - frost",
        },
    ];

    const INTERESTING_POINTS_SINE: [InterestingPoint; 8] = [
        InterestingPoint {
            coords: [1.0f32, 0f32],
            desc: "(1.0 + 0 * i)",
        },
        InterestingPoint {
            coords: [1.0f32, 0.1f32],
            desc: "(1.0 + 0.1f32 * i)",
        },
        InterestingPoint {
            coords: [1f32, 1f32],
            desc: "(1.0 + 1 * i)",
        },
        InterestingPoint {
            coords: [0.984808f32, 0.173648f32],
            desc: "(0.984808 + 0.173648 * i)",
        },
        InterestingPoint {
            coords: [-1.29904f32, -0.75f32],
            desc: "(-1.29904 + -0.75 * i)",
        },
        InterestingPoint {
            coords: [1.17462f32, 0.427525f32],
            desc: "(1.17462, 0.427525 * i)",
        },
        InterestingPoint {
            coords: [1.87939f32, 0.68404f32],
            desc: "(1.87939 + 0.68404 * i)",
        },
        InterestingPoint {
            coords: [-0.2f32, 1f32],
            desc: "(-0.2 + 1 * i)",
        },
    ];

    const INTERESTING_POINTS_COSINE: [InterestingPoint; 5] = [
        InterestingPoint {
            coords: [1.0f32, -0.5f32],
            desc: "(1.0 - 0.5 * i) - good colors",
        },
        InterestingPoint {
            coords: [
                std::f32::consts::FRAC_PI_2,
                std::f32::consts::FRAC_PI_2 * 0.6f32,
            ],
            desc: "(PI/2, PI/2 * 0.6) - dendrites",
        },
        InterestingPoint {
            coords: [
                std::f32::consts::FRAC_PI_2,
                std::f32::consts::FRAC_PI_2 * 0.4f32,
            ],
            desc: "(PI/2, PI/2 * 0.4) - dendrites",
        },
        InterestingPoint {
            coords: [
                std::f32::consts::FRAC_PI_2 * 2.0f32,
                std::f32::consts::FRAC_PI_2 * 0.25f32,
            ],
            desc: "(PI, PI/2 * 0.25) - fuzzy spots",
        },
        InterestingPoint {
            coords: [
                std::f32::consts::FRAC_PI_2 * 1.5f32,
                std::f32::consts::FRAC_PI_2 * 0.05f32,
            ],
            desc: "(PI * 1.5, PI/2 * 0.05) - fuzzy spots",
        },
    ];

    const INTERESTING_POINTS_CUBIC: [InterestingPoint; 5] = [
        InterestingPoint {
            coords: [-0.5f32, 0.05f32],
            desc: "-(0.5 + 0.05 * i)",
        },
        InterestingPoint {
            coords: [0.7f32, 0.5f32],
            desc: "(0.7 + 0.5 * i)",
        },
        InterestingPoint {
            coords: [0.53f32, 0.1f32],
            desc: "(0.53 + 0.1 * i)",
        },
        InterestingPoint {
            coords: [0.52f32, 0.1f32],
            desc: "(0.52 + 0.1 * i)",
        },
        InterestingPoint {
            coords: [0.515f32, 0.1f32],
            desc: "(0.515 + 0.1 * i)",
        },
    ];

    pub fn new(vks: &mut VulkanState, bindless: &mut BindlessResourceSystem) -> Julia {
        Self {
            params: JuliaCPU2GPU::new(
                Self::INTERESTING_POINTS_QUADRATIC[0].coords[0],
                Self::INTERESTING_POINTS_QUADRATIC[0].coords[1],
                JuliaIterationType::Quadratic,
            ),
            gpu_state: FractalGPUState::new(vks, bindless),
            point_quadratic_idx: 0,
            point_sine_idx: 0,
            point_cosine_idx: 0,
            point_cubic_idx: 0,
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
        self.gpu_state.render(vks, context, &self.params);
    }

    pub fn do_ui(&mut self, ui: &mut imgui::Ui) {
        ui.window("Julia fractal parameters")
            .always_auto_resize(true)
            .build(|| {
                self.params.core.do_ui(ui);

                ui.separator();
                let mut c_ptr: [f32; 2] = [self.params.c_x, self.params.c_y];

                if ui.input_float2("C point", &mut c_ptr).build() {
                    self.params.c_x = c_ptr[0];
                    self.params.c_y = c_ptr[1];
                }

                ui.separator();
                ui.text("Iteration:");

                enum_iterator::all::<JuliaIterationType>().for_each(|it| {
                    if ui.radio_button_bool(format!("{:?}", it), self.params.iteration == it) {
                        self.params.iteration = it;

                        //
                        // reset center + zoom
                        let center = match it {
                            JuliaIterationType::Quadratic => {
                                &Self::INTERESTING_POINTS_QUADRATIC[self.point_quadratic_idx].coords
                            }
                            JuliaIterationType::Cosine => {
                                &Self::INTERESTING_POINTS_COSINE[self.point_cosine_idx].coords
                            }
                            JuliaIterationType::Sine => {
                                &Self::INTERESTING_POINTS_SINE[self.point_sine_idx].coords
                            }
                            JuliaIterationType::Cubic => {
                                &Self::INTERESTING_POINTS_CUBIC[self.point_cubic_idx].coords
                            }
                        };

                        self.params.c_x = center[0];
                        self.params.c_y = center[1];
                        self.params.core.reset();
                    }
                });

                let do_combo_interest_points =
                    |label: &str, sel_idx: usize, points: &[InterestingPoint]| {
                        let mut pt_idx = sel_idx;

                        if ui.combo(label, &mut pt_idx, points, |pt_coords| {
                            std::borrow::Cow::Borrowed(&pt_coords.desc)
                        }) {
                            Some(pt_idx)
                        } else {
                            None
                        }
                    };

                ui.separator();
                match self.params.iteration {
                    JuliaIterationType::Quadratic => {
                        do_combo_interest_points(
                            "Interesting points to explore (quadratic)",
                            self.point_quadratic_idx,
                            &Self::INTERESTING_POINTS_QUADRATIC,
                        )
                        .map(|sel_idx| {
                            self.point_quadratic_idx = sel_idx;
                            self.params.c_x = Self::INTERESTING_POINTS_QUADRATIC[sel_idx].coords[0];
                            self.params.c_y = Self::INTERESTING_POINTS_QUADRATIC[sel_idx].coords[1];
                        });
                    }

                    JuliaIterationType::Sine => {
                        do_combo_interest_points(
                            "Interesting points to explore (sine)",
                            self.point_sine_idx,
                            &Self::INTERESTING_POINTS_SINE,
                        )
                        .map(|sel_idx| {
                            self.point_sine_idx = sel_idx;
                            self.params.c_x = Self::INTERESTING_POINTS_SINE[sel_idx].coords[0];
                            self.params.c_y = Self::INTERESTING_POINTS_SINE[sel_idx].coords[1];
                        });
                    }

                    JuliaIterationType::Cosine => {
                        do_combo_interest_points(
                            "Interesting points to explore (cosine)",
                            self.point_cosine_idx,
                            &Self::INTERESTING_POINTS_COSINE,
                        )
                        .map(|sel_idx| {
                            self.point_cosine_idx = sel_idx;
                            self.params.c_x = Self::INTERESTING_POINTS_COSINE[sel_idx].coords[0];
                            self.params.c_y = Self::INTERESTING_POINTS_COSINE[sel_idx].coords[1];
                        });
                    }

                    JuliaIterationType::Cubic => {
                        do_combo_interest_points(
                            "Interesting points to explore (cubic)",
                            self.point_cubic_idx,
                            &Self::INTERESTING_POINTS_CUBIC,
                        )
                        .map(|sel_idx| {
                            self.point_cubic_idx = sel_idx;
                            self.params.c_x = Self::INTERESTING_POINTS_CUBIC[sel_idx].coords[0];
                            self.params.c_y = Self::INTERESTING_POINTS_CUBIC[sel_idx].coords[1];
                        });
                    }
                }
            });
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, enum_iterator::Sequence)]
#[repr(u32)]
enum MandelbrotType {
    Standard,
    BurningShip,
}

#[derive(Copy, Clone, Debug)]
#[repr(C, align(16))]
struct MandelbrotCPU2GPU {
    core: FractalCore<MandelbrotCPU2GPU>,
    ftype: MandelbrotType,
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
    const BINDLESS_FRAGMENT_SHADER_MODULE: &'static str = "data/shaders/mandelbrot.bindless.frag";
}

pub struct Mandelbrot {
    params: MandelbrotCPU2GPU,
    gpu_state: FractalGPUState<MandelbrotCPU2GPU>,
}

impl Mandelbrot {
    pub fn new(vks: &mut VulkanState, bindless: &mut BindlessResourceSystem) -> Mandelbrot {
        Self {
            params: MandelbrotCPU2GPU {
                core: FractalCore::default(),
                ftype: MandelbrotType::Standard,
            },
            gpu_state: FractalGPUState::new(vks, bindless),
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
        self.gpu_state.render(vks, context, &self.params);
    }

    pub fn do_ui(&mut self, ui: &mut imgui::Ui) {
        ui.window("Mandelbrot fractal parameters")
            .always_auto_resize(true)
            .build(|| {
                enum_iterator::all::<MandelbrotType>().for_each(|it| {
                    if ui.radio_button_bool(format!("{:?}", it), self.params.ftype == it) {
                        self.params.ftype = it;
                        self.params.core.reset();
                    }
                });

                self.params.core.do_ui(ui);
            });
    }
}

struct FractalGPUState<T: FractalCoreParams + Copy> {
    pipeline: Pipeline,
    bindless_layout: PipelineLayout,
    ubo_params: UniqueBuffer,
    ubo_handle: BindlessResourceHandle,
    palettes: UniqueImage,
    palettes_view: UniqueImageView,
    sampler: UniqueSampler,
    palette_handle: BindlessResourceHandle,
    _marker: std::marker::PhantomData<T>,
}

impl<T> FractalGPUState<T>
where
    T: FractalCoreParams + Copy,
{
    fn make_palettes(vks: &mut VulkanState) -> (UniqueImage, UniqueImageView, UniqueSampler) {
        use enterpolation::{linear::ConstEquidistantLinear, Curve};
        use palette::{rgb, LinSrgb, Srgb};

        let gradient = ConstEquidistantLinear::<f32, _, 3>::equidistant_unchecked([
            LinSrgb::new(0.00, 0.05, 0.20),
            LinSrgb::new(0.70, 0.10, 0.20),
            LinSrgb::new(0.95, 0.90, 0.30),
        ]);

        let colors = gradient
            .take(512)
            .map(|c| {
                let a: Srgb<u8> = c.into();
                a.into_u32::<rgb::channels::Rgba>()
                //
            })
            .collect::<Vec<_>>();

        use ash::vk::ImageCreateInfo;
        let palette = UniqueImage::from_bytes(
            vks,
            *ImageCreateInfo::builder()
                .usage(ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST)
                .tiling(ImageTiling::OPTIMAL)
                .format(Format::R8G8B8A8_UNORM)
                .samples(SampleCountFlags::TYPE_1)
                .mip_levels(1)
                .array_layers(1)
                .extent(*Extent3D::builder().width(512).height(1).depth(1))
                .image_type(ImageType::TYPE_1D)
                .initial_layout(ImageLayout::UNDEFINED)
                .sharing_mode(SharingMode::EXCLUSIVE),
            unsafe { std::slice::from_raw_parts(colors.as_ptr() as *const u8, colors.len() * 4) },
        );

        let palette_view = UniqueImageView::new(
            vks,
            &palette,
            *ImageViewCreateInfo::builder()
                .format(ash::vk::Format::R8G8B8A8_UNORM)
                .image(palette.image)
                .subresource_range(
                    *ash::vk::ImageSubresourceRange::builder()
                        .aspect_mask(ash::vk::ImageAspectFlags::COLOR)
                        .base_array_layer(0)
                        .base_mip_level(0)
                        .layer_count(1)
                        .level_count(1),
                )
                .view_type(ash::vk::ImageViewType::TYPE_1D_ARRAY)
                .components(
                    *ash::vk::ComponentMapping::builder()
                        .a(ComponentSwizzle::IDENTITY)
                        .r(ComponentSwizzle::IDENTITY)
                        .g(ComponentSwizzle::IDENTITY)
                        .b(ComponentSwizzle::IDENTITY),
                ),
        );

        let sampler = UniqueSampler::new(
            vks,
            *SamplerCreateInfo::builder()
                .address_mode_u(SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(SamplerAddressMode::CLAMP_TO_EDGE)
                .border_color(BorderColor::INT_OPAQUE_BLACK)
                .mag_filter(Filter::LINEAR)
                .min_filter(Filter::LINEAR)
                .mipmap_mode(SamplerMipmapMode::LINEAR),
        );

        (palette, palette_view, sampler)
    }

    fn new(vks: &mut VulkanState, bindless: &mut BindlessResourceSystem) -> Self {
        let pipeline = Self::create_graphics_pipeline(&vks.ds, vks.renderpass, bindless);

        let ubo_params = UniqueBuffer::new::<T>(
            vks,
            BufferUsageFlags::STORAGE_BUFFER,
            MemoryPropertyFlags::DEVICE_LOCAL | MemoryPropertyFlags::HOST_VISIBLE,
            vks.swapchain.max_frames as usize,
        );

        let ubo_handle = bindless.register_ssbo(&vks.ds, &ubo_params);
        let (palettes, palettes_view, sampler) = Self::make_palettes(vks);
        let palette_handle = bindless.register_image(&vks.ds, &palettes_view, &sampler);

        Self {
            pipeline,
            bindless_layout: bindless.bindless_pipeline_layout(),
            ubo_params,
            ubo_handle,
            palettes,
            palettes_view,
            sampler,
            palette_handle,
            _marker: std::marker::PhantomData::<T>,
        }
    }

    fn render(&mut self, vks: &VulkanState, context: &FrameRenderContext, gpu_data: &T) {
        let render_area = Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent: context.fb_size,
        };

        // let data = T {
        //     palette: self.palette_handle.get_id(),
        //     ..*gpu_data
        // };
        let mut data = *gpu_data;
        data.palette = 0;

        //
        // copy params
        UniqueBufferMapping::new(
            &self.ubo_params,
            &vks.ds,
            Some(size_of::<T>() * context.current_frame_id as usize),
            Some(size_of::<T>()),
        )
        .write_data(std::slice::from_ref(gpu_data));

        unsafe {
            vks.ds.device.cmd_set_viewport(
                context.cmd_buff,
                0,
                &[Viewport {
                    x: 0f32,
                    y: context.fb_size.height as f32,
                    width: context.fb_size.width as f32,
                    height: -(context.fb_size.height as f32),
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

            let id = BindlessResourceSystem::make_push_constant(
                context.current_frame_id,
                self.ubo_handle,
            );

            vks.ds.device.cmd_push_constants(
                context.cmd_buff,
                self.bindless_layout,
                ShaderStageFlags::ALL,
                0,
                &id.to_le_bytes(),
            );

            vks.ds.device.cmd_draw(context.cmd_buff, 3, 1, 0, 0);
        }
    }

    fn create_graphics_pipeline(
        ds: &VulkanDeviceState,
        renderpass: RenderPass,
        bindless: &BindlessResourceSystem,
    ) -> Pipeline {
        let vsm = compile_shader_from_file(T::VERTEX_SHADER_MODULE, &ds.device).unwrap();
        let fsm = compile_shader_from_file(T::BINDLESS_FRAGMENT_SHADER_MODULE, &ds.device).unwrap();

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
                            .vertex_attribute_descriptions(&[])
                            .vertex_binding_descriptions(&[]),
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
                    .layout(bindless.bindless_pipeline_layout())
                    .render_pass(renderpass)
                    .subpass(0)],
                None,
            )
        }
        .expect("Failed to create graphics pipeline ... ");

        pipeline[0]
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
