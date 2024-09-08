use std::mem::size_of;

use ash::vk::{
    BufferUsageFlags, ClearColorValue, ClearValue, MemoryPropertyFlags, Offset2D,
    PipelineBindPoint, Rect2D, RenderPassBeginInfo, SubpassContents,
};

use fractal::{Julia, Mandelbrot};
use ui::UiBackend;
use vulkan_renderer::{
    BindlessResourceSystem, FrameRenderContext, UniqueBuffer, UniqueBufferMapping, VulkanState,
};
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

mod fractal;
mod shader;
mod ui;
mod vulkan_renderer;

use enum_iterator::{next_cycle, previous_cycle};

fn main() {
    let _logger = flexi_logger::Logger::with(
        flexi_logger::LogSpecification::builder()
            .default(flexi_logger::LevelFilter::Debug)
            .build(),
    )
    .adaptive_format_for_stderr(flexi_logger::AdaptiveFormat::Detailed)
    .log_to_file(flexi_logger::FileSpec::default())
    .write_mode(flexi_logger::WriteMode::BufferAndFlush)
    .duplicate_to_stderr(flexi_logger::Duplicate::All)
    .start()
    .unwrap_or_else(|e| {
        panic!("Failed to start the logger {}", e);
    });

    use crate::shader::*;
    let _ = compile_shader(&ShaderCompileInfo {
        src: ShaderSource::File("data/shaders/apps/triangle/tri.vert".into()),
        entry_point: None,
        optimize: false,
        debug_info: false,
        compile_defs: &[("FRAME_BASED_SHADER", None)],
    })
    .expect("Sum tin wong");
    // let spv_code = include_bytes!("test.spv");
    // crate::shader::reflect_shader_module(&spv_code);

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
    bindless_sys: BindlessResourceSystem,
    ubo_globals: UniqueBuffer,
}

#[cfg(target_os = "windows")]
fn get_window_data(win: &winit::window::Window) -> WindowSystemIntegration {
    use winit::platform::windows::WindowExtWindows;

    WindowSystemIntegration {
        hwnd: win.hwnd(),
        hinstance: win.hinstance(),
    }
}

#[cfg(target_os = "linux")]
fn get_window_data(win: &winit::window::Window) -> vulkan_renderer::WindowSystemIntegration {
    use winit::platform::x11::WindowExtX11;
    vulkan_renderer::WindowSystemIntegration {
        native_disp: win.xlib_display().unwrap(),
        native_win: win.xlib_window().unwrap(),
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
struct UniformGlobals {
    // mat: [f32; 16],
    frame_id: u32,
    // pad: [u32; 3],
}

impl FractalSimulation {
    fn new(window: &winit::window::Window) -> FractalSimulation {
        let cursor_pos = (
            (window.inner_size().width / 2) as f32,
            (window.inner_size().height / 2) as f32,
        );

        log::info!("Cursor initial position {:?}", cursor_pos);

        let mut vks = Box::pin(
            VulkanState::new(get_window_data(window), window.inner_size().into())
                .expect("Failed to initialize vulkan ..."),
        );

        log::info!("### Device limits:\n{:?}", vks.limits());
        log::info!("### Device features:\n{:?}", vks.features());

        vks.begin_resource_loading();

        let mut bindless_sys = BindlessResourceSystem::new(&vks);

        let ui = UiBackend::new(window, &mut vks, &mut bindless_sys, ui::HiDpiMode::Default);
        let mandelbrot = Mandelbrot::new(&mut vks, &mut bindless_sys);
        let julia = Julia::new(&mut vks, &mut bindless_sys);

        vks.end_resource_loading();

        let ubo_globals = UniqueBuffer::new::<UniformGlobals>(
            &vks,
            BufferUsageFlags::UNIFORM_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            vks.swapchain.max_frames as usize,
        );
        bindless_sys.register_uniform_buffer(&vks.ds, &ubo_globals);

        FractalSimulation {
            ftype: FractalType::Mandelbrot,
            julia,
            mandelbrot,
            ui_opened: true,
            control_down: false,
            cursor_pos,
            vks,
            ui,
            ubo_globals,
            bindless_sys,
        }
    }

    fn begin_rendering(&mut self, window: &winit::window::Window) -> FrameRenderContext {
        let img_size = ash::vk::Extent2D {
            width: window.inner_size().width,
            height: window.inner_size().height,
        };

        let frame_context = self.vks.begin_rendering(img_size);

        let render_area = Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent: frame_context.fb_size,
        };

        //log::info!("img_size {:?}, render area {:?}", img_size, render_area);

        let ubo_data = UniformGlobals {
            // mat: [1f32; 16],
            frame_id: frame_context.current_frame_id,
            // pad: [0u32; 3],
        };

        UniqueBufferMapping::new(
            &self.ubo_globals,
            &self.vks.ds,
            Some(size_of::<UniformGlobals>() * frame_context.current_frame_id as usize),
            Some(size_of::<UniformGlobals>()),
        )
        .write_data(std::slice::from_ref(&ubo_data));

        unsafe {
            self.vks.ds.device.cmd_bind_descriptor_sets(
                frame_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.bindless_sys.bindless_pipeline_layout(),
                0,
                self.bindless_sys.descriptor_sets(),
                &[],
            );
        }

        frame_context
    }

    fn setup_ui(&mut self, window: &winit::window::Window) {
        let ui = self.ui.new_frame(window);

        ui.window("Fractal type")
            .size([800f32, 600f32], imgui::Condition::Always)
            .resizable(true)
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
                    });

                ui.separator();
                ui.text_colored([0.0f32, 1.0f32, 0.0f32, 1.0f32], "::: Parameters :::");
                ui.separator();

                match self.ftype {
                    FractalType::Julia => self.julia.do_ui(ui),
                    FractalType::Mandelbrot => self.mandelbrot.do_ui(ui),
                }
            });
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
                let frame_context = self.begin_rendering(window);

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

                self.vks.end_rendering(frame_context);
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
