use ash::vk::{
    BorderColor, BufferUsageFlags, CullModeFlags, DynamicState, Filter, Format, FrontFace,
    ImageType, ImageUsageFlags, MemoryPropertyFlags, Offset2D, PipelineBindPoint,
    PipelineDepthStencilStateCreateInfo, PipelineRasterizationStateCreateInfo, PolygonMode, Rect2D,
    SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, ShaderStageFlags, Viewport,
};
use enum_iterator::{next_cycle, previous_cycle};

use crate::{
    vulkan_bindless::{
        BindlessImageResourceHandleEntryPair, BindlessResourceSystem,
        BindlessStorageBufferResourceHandleEntryPair, GlobalPushConstant,
    },
    vulkan_buffer::{VulkanBuffer, VulkanBufferCreateInfo},
    vulkan_image::{UniqueImage, UniqueSampler, VulkanImageCreateInfo},
    vulkan_mapped_memory::UniqueBufferMapping,
    vulkan_pipeline::{GraphicsPipelineCreateOptions, GraphicsPipelineSetupHelper, UniquePipeline},
    vulkan_renderer::{FrameRenderContext, GraphicsError, QueueType, VulkanRenderer},
    vulkan_shader::ShaderSource,
    InputState,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq, enum_iterator::Sequence, num_enum::FromPrimitive)]
#[repr(u32)]
pub enum Coloring {
    #[default]
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

    fn ssbo_size() -> usize;
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct FractalCommonCore {
    screen_width: u32,
    screen_height: u32,
    iterations: u32,
    zoom: f32,
    ox: f32,
    oy: f32,
    coloring: u32,
    fxmin: f32,
    fxmax: f32,
    fymin: f32,
    fymax: f32,
    escape_radius: u32,
    palette_handle: u32,
    palette_idx: u32,
}

impl FractalCommonCore {
    fn new<T: FractalCoreParams>(palette_handle: u32) -> Self {
        Self {
            screen_width: 1024,
            screen_height: 1024,
            iterations: 32,
            zoom: 1f32,
            ox: 0f32,
            oy: 0f32,
            coloring: Coloring::BlackWhite as u32,
            fxmin: -T::FRACTAL_HALF_WIDTH,
            fxmax: T::FRACTAL_HALF_WIDTH,
            fymin: -T::FRACTAL_HALF_HEIGHT,
            fymax: T::FRACTAL_HALF_HEIGHT,
            escape_radius: T::ESC_RADIUS_MIN,
            palette_handle,
            palette_idx: 0u32,
        }
    }

    pub fn input_handler<T: FractalCoreParams>(&mut self, input_state: &InputState) {
        use winit::event::ElementState;
        use winit::event::MouseButton;
        use winit::event::VirtualKeyCode;
        use winit::event::WindowEvent;

        match *input_state.event {
            WindowEvent::Resized(new_size) => {
                self.screen_resized::<T>(new_size.width, new_size.height);
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
                    self.zoom_out::<T>();
                } else {
                    self.zoom_in::<T>();
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
                    self.next_color::<T>();
                }

                Some(VirtualKeyCode::NumpadSubtract) => {
                    self.decrease_iterations::<T>();
                }

                Some(VirtualKeyCode::NumpadAdd) => {
                    self.increase_iterations::<T>();
                }

                Some(VirtualKeyCode::Insert) => {
                    self.increase_escape_radius::<T>();
                }

                Some(VirtualKeyCode::Delete) => {
                    self.decrease_escape_radius::<T>();
                }
                _ => {}
            },

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => match button {
                MouseButton::Left => {
                    self.center_moved::<T>(
                        input_state.cursor_pos.0,
                        input_state.cursor_pos.1,
                        input_state.control_down,
                    );
                }
                MouseButton::Right => {
                    self.zoom_out::<T>();
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn center_moved<T: FractalCoreParams>(&mut self, cx: f32, cy: f32, zoom: bool) {
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

    fn zoom_out<T: FractalCoreParams>(&mut self) {
        self.zoom = (self.zoom * T::ZOOM_OUT_FACTOR).min(1f32);
        self.fxmin = self.ox - T::FRACTAL_HALF_WIDTH * self.zoom;
        self.fxmax = self.ox + T::FRACTAL_HALF_WIDTH * self.zoom;
        self.fymin = self.oy - T::FRACTAL_HALF_HEIGHT * self.zoom;
        self.fymax = self.oy + T::FRACTAL_HALF_HEIGHT * self.zoom;
    }

    fn zoom_in<T: FractalCoreParams>(&mut self) {
        self.zoom = (self.zoom * T::ZOOM_IN_FACTOR).max(0f32);
        self.fxmin = self.ox - T::FRACTAL_HALF_WIDTH * self.zoom;
        self.fxmax = self.ox + T::FRACTAL_HALF_WIDTH * self.zoom;
        self.fymin = self.oy - T::FRACTAL_HALF_HEIGHT * self.zoom;
        self.fymax = self.oy + T::FRACTAL_HALF_HEIGHT * self.zoom;
    }

    fn screen_resized<T: FractalCoreParams>(&mut self, width: u32, height: u32) {
        self.screen_width = width;
        self.screen_height = height;
    }

    fn increase_iterations<T: FractalCoreParams>(&mut self) {
        self.iterations = (self.iterations * 2).min(T::MAX_ITERATIONS);
    }

    fn decrease_iterations<T: FractalCoreParams>(&mut self) {
        self.iterations = (self.iterations / 2).max(T::MIN_ITERATIONS);
    }

    fn next_color<T: FractalCoreParams>(&mut self) {
        self.coloring = next_cycle(&Coloring::from(self.coloring)).unwrap() as _;
    }

    fn previous_color(&mut self) {
        self.coloring = previous_cycle(&Coloring::from(self.coloring)).unwrap() as _;
    }

    fn increase_escape_radius<T: FractalCoreParams>(&mut self) {
        self.escape_radius = (self.escape_radius * 2).min(T::ESC_RADIUS_MAX);
    }

    fn decrease_escape_radius<T: FractalCoreParams>(&mut self) {
        self.escape_radius = (self.escape_radius / 2).max(T::ESC_RADIUS_MIN);
    }

    fn reset<T: FractalCoreParams>(&mut self) {
        *self = Self {
            screen_width: self.screen_width,
            screen_height: self.screen_height,
            palette_idx: self.palette_idx,
            ..Self::new::<T>(self.palette_handle)
        };
    }

    pub fn do_ui<T: FractalCoreParams>(&mut self, ui: &imgui::Ui) {
        //
        // coloring algorithm
        if let Some(_cb) = ui.begin_combo(
            "Coloring algorithm",
            format!("{:?}", Coloring::from(self.coloring)),
        ) {
            let mut selected = Coloring::from(self.coloring);
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
                    self.coloring = item as _;
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
        ui.text_colored([0.0f32, 1.0f32, 0.0f32, 1.0f32], "::: Info :::");
        ui.separator();

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
            format!(
                "Cursor position: ({:.2}, {:.2})",
                cursor_pos[0], cursor_pos[1]
            ),
        );

        ui.text_colored(
            [1f32, 0f32, 0f32, 1f32],
            format!("Center: ({:.4}, {:.4})", self.ox, self.oy),
        );
        ui.text_colored(
            [1f32, 0f32, 0f32, 1f32],
            format!(
                "Domain: ({:.4}, {:.4}) x ({:.4}, {:.4})",
                self.fxmin, self.fymin, self.fxmax, self.fymax
            ),
        );
        ui.text_colored(
            [1f32, 0f32, 0f32, 1f32],
            format!("Zoom: {}x", 1f32 / self.zoom),
        );
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, enum_iterator::Sequence, num_enum::FromPrimitive)]
#[repr(u32)]
enum JuliaIterationType {
    #[default]
    Quadratic,
    Sine,
    Cosine,
    Cubic,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct JuliaCPU2GPU {
    core: FractalCommonCore,
    c_x: f32,
    c_y: f32,
    iteration: u32,
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

    fn ssbo_size() -> usize {
        log::info!("Julia SSBO item size {}", std::mem::size_of::<Self>());
        std::mem::size_of::<Self>()
    }
}

impl JuliaCPU2GPU {
    fn new(c_x: f32, c_y: f32, iteration: JuliaIterationType, palette_handle: u32) -> JuliaCPU2GPU {
        Self {
            core: FractalCommonCore::new::<JuliaCPU2GPU>(palette_handle),
            c_x,
            c_y,
            iteration: iteration as _,
        }
    }
}

pub struct Julia {
    params: JuliaCPU2GPU,
    gpu_state: FractalGPUState,
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

    pub fn create(
        vks: &mut VulkanRenderer,
        bindless: &mut BindlessResourceSystem,
    ) -> std::result::Result<Julia, GraphicsError> {
        let gpu_state = FractalGPUState::new::<JuliaCPU2GPU>(vks, bindless)?;

        Ok(Self {
            params: JuliaCPU2GPU::new(
                Self::INTERESTING_POINTS_QUADRATIC[0].coords[0],
                Self::INTERESTING_POINTS_QUADRATIC[0].coords[1],
                JuliaIterationType::Quadratic,
                gpu_state.palettes.0.handle(),
            ),
            gpu_state,
            point_quadratic_idx: 0,
            point_sine_idx: 0,
            point_cosine_idx: 0,
            point_cubic_idx: 0,
        })
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
                self.params.core.reset::<JuliaCPU2GPU>();
            }

            _ => self.params.core.input_handler::<JuliaCPU2GPU>(input_state),
        }
    }

    pub fn render(&mut self, vks: &VulkanRenderer, context: &FrameRenderContext) {
        self.gpu_state
            .render(vks, context, std::slice::from_ref(&self.params));
    }

    pub fn do_ui(&mut self, ui: &imgui::Ui) {
        self.params.core.do_ui::<JuliaCPU2GPU>(ui);

        ui.separator();
        let mut c_ptr: [f32; 2] = [self.params.c_x, self.params.c_y];

        if ui.input_float2("C point", &mut c_ptr).build() {
            self.params.c_x = c_ptr[0];
            self.params.c_y = c_ptr[1];
        }

        ui.separator();
        ui.text("Iteration:");

        enum_iterator::all::<JuliaIterationType>().for_each(|it| {
            if ui.radio_button_bool(
                format!("{:?}", it),
                JuliaIterationType::from(self.params.iteration) == it,
            ) {
                self.params.iteration = it as _;

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
                self.params.core.reset::<JuliaCPU2GPU>();
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
        match JuliaIterationType::from(self.params.iteration) {
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
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, enum_iterator::Sequence, num_enum::FromPrimitive)]
#[repr(u32)]
enum MandelbrotType {
    #[default]
    Standard,
    BurningShip,
}

#[derive(Copy, Clone, Debug)]
struct MandelbrotCPU2GPU {
    core: FractalCommonCore,
    ftype: u32,
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

    fn ssbo_size() -> usize {
        log::info!(
            "Mandelbrot SSBO item length {}",
            std::mem::size_of::<Self>()
        );
        std::mem::size_of::<Self>()
    }
}

pub struct Mandelbrot {
    params: MandelbrotCPU2GPU,
    gpu_state: FractalGPUState,
}

impl Mandelbrot {
    pub fn new(
        vks: &mut VulkanRenderer,
        bindless: &mut BindlessResourceSystem,
    ) -> std::result::Result<Mandelbrot, GraphicsError> {
        let gpu_state = FractalGPUState::new::<MandelbrotCPU2GPU>(vks, bindless)?;

        Ok(Self {
            params: MandelbrotCPU2GPU {
                core: FractalCommonCore::new::<MandelbrotCPU2GPU>(gpu_state.palettes.0.handle()),
                ftype: MandelbrotType::Standard as _,
            },
            gpu_state,
        })
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
                self.params.core.reset::<MandelbrotCPU2GPU>();
            }

            _ => self
                .params
                .core
                .input_handler::<MandelbrotCPU2GPU>(input_state),
        }
    }

    pub fn render(&mut self, vks: &VulkanRenderer, context: &FrameRenderContext) {
        self.gpu_state
            .render(vks, context, std::slice::from_ref(&self.params));
    }

    pub fn do_ui(&mut self, ui: &imgui::Ui) {
        enum_iterator::all::<MandelbrotType>().for_each(|it| {
            if ui.radio_button_bool(
                format!("{:?}", it),
                MandelbrotType::from(self.params.ftype) == it,
            ) {
                self.params.ftype = it as _;
                self.params.core.reset::<MandelbrotCPU2GPU>();
            }
        });

        self.params.core.do_ui::<MandelbrotCPU2GPU>(ui);
    }
}

struct FractalGPUState {
    pipeline: UniquePipeline,
    ubo_handle: BindlessStorageBufferResourceHandleEntryPair,
    palettes: BindlessImageResourceHandleEntryPair,
    sampler: UniqueSampler,
}

impl FractalGPUState {
    fn make_palettes(
        vks: &mut VulkanRenderer,
        bindless_sys: &mut BindlessResourceSystem,
    ) -> std::result::Result<(BindlessImageResourceHandleEntryPair, UniqueSampler), GraphicsError>
    {
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
                a.into_u32::<rgb::channels::Abgr>()
            })
            .collect::<Vec<_>>();

        let pixels =
            unsafe { std::slice::from_raw_parts(colors.as_ptr() as *const u8, colors.len() * 4) };
        crate::vulkan_renderer::misc::write_ppm("color.palette.ppm", 512, 1, pixels);

        let qjob = vks.create_queue_job(QueueType::Transfer).unwrap();
        let palette = UniqueImage::from_bytes(
            vks,
            &VulkanImageCreateInfo {
                tag_name: Some("UI color palette"),
                work_pkg: Some(&qjob),
                ty: ImageType::TYPE_2D,
                usage: ImageUsageFlags::SAMPLED,
                memory: MemoryPropertyFlags::DEVICE_LOCAL,
                format: Format::R8G8B8A8_UNORM,
                cubemap: false,
                width: 512,
                height: 1,
                depth: 1,
                layers: 1,
                pixels: &[unsafe {
                    std::slice::from_raw_parts(colors.as_ptr() as *const u8, colors.len() * 4)
                }],
            },
        )?;

        let wait_token = vks.submit_queue_job(qjob)?;
        let sampler = UniqueSampler::new(
            vks,
            SamplerCreateInfo::default()
                .address_mode_u(SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(SamplerAddressMode::CLAMP_TO_EDGE)
                .border_color(BorderColor::INT_OPAQUE_BLACK)
                .mag_filter(Filter::LINEAR)
                .min_filter(Filter::LINEAR)
                .mipmap_mode(SamplerMipmapMode::LINEAR),
        )?;

        let color_palette = bindless_sys.register_image(palette, &sampler, None);
        log::info!("color palette: {color_palette:?}");
        vks.consume_wait_token(wait_token);
        Ok((color_palette, sampler))
    }

    fn new<T: FractalCoreParams>(
        renderer: &mut VulkanRenderer,
        bindless: &mut BindlessResourceSystem,
    ) -> std::result::Result<Self, GraphicsError> {
        let pipeline = GraphicsPipelineSetupHelper::new()
            .add_shader_stage(ShaderSource::File(T::VERTEX_SHADER_MODULE.into()))
            .add_shader_stage(ShaderSource::File(
                T::BINDLESS_FRAGMENT_SHADER_MODULE.into(),
            ))
            .set_depth_stencil_state(
                PipelineDepthStencilStateCreateInfo::default()
                    .depth_test_enable(false)
                    .depth_write_enable(false)
                    .stencil_test_enable(false)
                    .min_depth_bounds(0f32)
                    .max_depth_bounds(1f32),
            )
            .set_raster_state(
                PipelineRasterizationStateCreateInfo::default()
                    .cull_mode(CullModeFlags::NONE)
                    .front_face(FrontFace::COUNTER_CLOCKWISE)
                    .polygon_mode(PolygonMode::FILL)
                    .line_width(1f32),
            )
            .set_dynamic_state(&[
                DynamicState::VIEWPORT_WITH_COUNT,
                DynamicState::SCISSOR_WITH_COUNT,
            ])
            .create(
                renderer,
                GraphicsPipelineCreateOptions {
                    layout: Some(bindless.pipeline_layout()),
                },
            )?;

        let ubo_params = VulkanBuffer::create(
            renderer,
            &VulkanBufferCreateInfo {
                name_tag: None,
                work_package: None,
                usage: BufferUsageFlags::STORAGE_BUFFER,
                memory_properties: MemoryPropertyFlags::DEVICE_LOCAL
                    | MemoryPropertyFlags::HOST_VISIBLE,
                slabs: renderer.max_frames() as usize,
                bytes: T::ssbo_size(),
                initial_data: &[],
            },
        )?;

        let ubo_handle = bindless.register_storage_buffer(ubo_params, None);
        let (palette_image, sampler) = Self::make_palettes(renderer, bindless)?;
        renderer.queue_ownership_transfer(&palette_image);

        log::info!("fractal ubo {ubo_handle:?}");

        Ok(Self {
            ubo_handle,
            pipeline,
            palettes: palette_image,
            sampler,
        })
    }

    fn render<T: Sized + Copy>(
        &mut self,
        vks: &VulkanRenderer,
        context: &FrameRenderContext,
        gpu_data: &[T],
    ) {
        let render_area = Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent: context.fb_size,
        };

        //
        // copy params

        let _ = UniqueBufferMapping::map_memory(
            vks.logical(),
            self.ubo_handle.1.devmem,
            self.ubo_handle.1.aligned_slab_size * context.current_frame_id as usize,
            self.ubo_handle.1.aligned_slab_size,
        )
        .map(|mapped_buffer| {
            mapped_buffer.write_data(gpu_data);
        });

        unsafe {
            vks.logical().cmd_set_viewport_with_count(
                context.cmd_buff,
                &[Viewport {
                    x: 0f32,
                    y: context.fb_size.height as f32,
                    width: context.fb_size.width as f32,
                    height: -(context.fb_size.height as f32),
                    min_depth: 0f32,
                    max_depth: 1f32,
                }],
            );

            vks.logical()
                .cmd_set_scissor_with_count(context.cmd_buff, &[render_area]);

            vks.logical().cmd_bind_pipeline(
                context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.handle(),
            );

            vks.logical().cmd_push_constants(
                context.cmd_buff,
                self.pipeline.layout(),
                ShaderStageFlags::ALL,
                0,
                &GlobalPushConstant::from_resource(
                    self.ubo_handle.0,
                    context.current_frame_id,
                    None,
                )
                .to_gpu(),
            );

            vks.logical().cmd_draw(context.cmd_buff, 3, 1, 0, 0);
        }
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
