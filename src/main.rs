use ash::vk::{
    ClearColorValue, ClearValue, Offset2D, Rect2D, RenderPassBeginInfo, SubpassContents,
};

use fractal::{Julia, Mandelbrot};
use ui::UiBackend;
use vulkan_renderer::{FrameRenderContext, UniqueBuffer, UniqueBufferMapping, VulkanState};
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

mod fractal;
mod ui;
mod vulkan_renderer;

use enum_iterator::{next_cycle, previous_cycle};

use crate::vulkan_renderer::WindowSystemIntegration;

fn main() {
    let _logger = flexi_logger::Logger::with(
        flexi_logger::LogSpecification::builder()
            .default(flexi_logger::LevelFilter::Debug)
            .build(),
    )
    .adaptive_format_for_stderr(flexi_logger::AdaptiveFormat::Detailed)
    .start()
    .unwrap_or_else(|e| {
        panic!("Failed to start the logger {}", e);
    });

    let event_loop = EventLoop::new();
    let primary_monitor = event_loop
        .primary_monitor()
        .expect("Failed to obtain primary monitor");

    let monitor_size = primary_monitor.size();
    log::info!("{:?}", primary_monitor);

    let window = WindowBuilder::new()
        .with_title("Fractal Explorer (with Rust + Vulkan)")
        .with_fullscreen(Some(winit::window::Fullscreen::Borderless(Some(
            primary_monitor,
        ))))
        .with_inner_size(monitor_size)
        .build(&event_loop)
        .unwrap();

    log::info!("Main window surface size {:?}", window.inner_size());

    window
        .set_cursor_position(PhysicalPosition::new(
            window.inner_size().width / 2,
            window.inner_size().height / 2,
        ))
        .expect("Failed to center cursor ...");

    let mut fractal_sim = FractalSimulation::new(&window);

    event_loop.run(move |event, _, control_flow| {
        fractal_sim.handle_event(&window, event, control_flow);
    });
}

pub struct InputState<'a> {
    pub window: &'a winit::window::Window,
    pub event: &'a winit::event::WindowEvent<'a>,
    pub control_down: bool,
    pub cursor_pos: (f32, f32),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, enum_iterator::Sequence)]
enum FractalType {
    Mandelbrot,
    Julia,
}

struct FractalSimulation {
    ftype: FractalType,
    julia: Julia,
    mandelbrot: Mandelbrot,
    ui_opened: bool,
    ui: UiBackend,
    control_down: bool,
    cursor_pos: (f32, f32),
    vks: std::pin::Pin<Box<VulkanState>>,
}

impl FractalSimulation {
    fn new(window: &winit::window::Window) -> FractalSimulation {
        let cursor_pos = (
            (window.inner_size().width / 2) as f32,
            (window.inner_size().height / 2) as f32,
        );

        use winit::platform::x11::WindowExtX11;
        let wsi = WindowSystemIntegration {
            native_disp: window.xlib_display().unwrap(),
            native_win: window.xlib_window().unwrap(),
        };

        log::info!("Cursor initial position {:?}", cursor_pos);

        let mut vks = Box::pin(VulkanState::new(wsi).expect("Failed to initialize vulkan ..."));
        vks.begin_resource_loading();

        let ui = UiBackend::new(window, &mut vks, ui::HiDpiMode::Default);
        let mandelbrot = Mandelbrot::new(&vks);
        let julia = Julia::new(&vks);

        vks.end_resource_loading();

        FractalSimulation {
            ftype: FractalType::Mandelbrot,
            julia,
            mandelbrot,
            ui_opened: true,
            control_down: false,
            cursor_pos,
            vks,
            ui,
        }
    }

    fn begin_rendering(&mut self) -> FrameRenderContext {
        let img_size = self.vks.ds.surface.image_size;
        let frame_context = self.vks.begin_rendering(img_size);

        let render_area = Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent: frame_context.fb_size,
        };

        unsafe {
            self.vks.ds.device.cmd_begin_render_pass(
                frame_context.cmd_buff,
                &RenderPassBeginInfo::builder()
                    .framebuffer(frame_context.framebuffer)
                    .render_area(render_area)
                    .render_pass(self.vks.renderpass)
                    .clear_values(&[ClearValue {
                        color: ClearColorValue {
                            float32: [0f32, 0f32, 0f32, 1f32],
                        },
                    }]),
                SubpassContents::INLINE,
            );
        }

        frame_context
    }

    fn end_rendering(&mut self, frame_context: &FrameRenderContext) {
        unsafe {
            self.vks
                .ds
                .device
                .cmd_end_render_pass(frame_context.cmd_buff);
        }

        self.vks.end_rendering();
    }

    fn setup_ui(&mut self, window: &winit::window::Window) {
        let ui = self.ui.new_frame(window);
        ui.window("Fractal type")
            .size([400f32, 100f32], imgui::Condition::Always)
            .build(|| {
                ui.begin_combo("Fractal type", format!("{:?}", self.ftype))
                    .map(|_| {
                        let mut selected = self.ftype;
                        for item in enum_iterator::all::<FractalType>() {
                            if selected == item {
                                ui.set_item_default_focus();
                            }

                            let clicked = ui
                                .selectable_config(format!("{:?}", item))
                                .selected(selected == item)
                                .build();

                            //
                            // When item is clicked, store it
                            if clicked {
                                selected = item;
                                self.ftype = item;
                            }
                        }
                    })
            });

        match self.ftype {
            FractalType::Julia => self.julia.do_ui(ui),
            FractalType::Mandelbrot => self.mandelbrot.do_ui(ui),
        }
    }

    fn handle_window_event(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
        control_flow: &mut winit::event_loop::ControlFlow,
    ) {
        match *event {
            WindowEvent::CloseRequested => {
                control_flow.set_exit();
            }

            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode:
                            vk @ Some(VirtualKeyCode::Up)
                            | vk @ Some(VirtualKeyCode::Escape)
                            | vk @ Some(VirtualKeyCode::Down),
                        ..
                    },
                ..
            } => match vk {
                Some(VirtualKeyCode::Escape) => control_flow.set_exit(),
                Some(VirtualKeyCode::Up) => self.ftype = next_cycle(&self.ftype).unwrap(),
                Some(VirtualKeyCode::Down) => self.ftype = previous_cycle(&self.ftype).unwrap(),
                _ => {}
            },

            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos.0 = position.x as f32;
                self.cursor_pos.1 = position.y as f32;
            }

            WindowEvent::ModifiersChanged(mods) => {
                self.control_down = mods.ctrl();
            }

            WindowEvent::Resized(_) => {
                //
                // forward this event to both fractal objects
                self.julia.input_handler(&InputState {
                    window,
                    event,
                    cursor_pos: self.cursor_pos,
                    control_down: self.control_down,
                });
                self.mandelbrot.input_handler(&InputState {
                    window,
                    event,
                    cursor_pos: self.cursor_pos,
                    control_down: self.control_down,
                });
            }

            _ => match self.ftype {
                FractalType::Julia => {
                    self.julia.input_handler(&InputState {
                        window,
                        event,
                        cursor_pos: self.cursor_pos,
                        control_down: self.control_down,
                    });
                }
                FractalType::Mandelbrot => {
                    self.mandelbrot.input_handler(&InputState {
                        window,
                        event,
                        cursor_pos: self.cursor_pos,
                        control_down: self.control_down,
                    });
                }
            },
        }
    }

    fn handle_event(
        &mut self,
        window: &winit::window::Window,
        event: Event<()>,
        control_flow: &mut winit::event_loop::ControlFlow,
    ) {
        control_flow.set_poll();

        match event {
            Event::WindowEvent {
                event: ref win_event,
                ..
            } => {
                let wants_input = self.ui.handle_event(window, &event);
                if !wants_input {
                    self.handle_window_event(window, win_event, control_flow);
                }
            }

            Event::MainEventsCleared => {
                let frame_context = self.begin_rendering();

                match self.ftype {
                    FractalType::Julia => {
                        self.julia.render(&self.vks, &frame_context);
                    }
                    FractalType::Mandelbrot => {
                        self.mandelbrot.render(&self.vks, &frame_context);
                    }
                }

                self.setup_ui(window);
                self.ui.draw_frame(&self.vks, &frame_context);

                self.end_rendering(&frame_context);
                std::thread::sleep(std::time::Duration::from_millis(20));
            }

            _ => {}
        }
    }
}

impl std::ops::Drop for FractalSimulation {
    fn drop(&mut self) {
        self.vks.wait_all_idle();
    }
}
