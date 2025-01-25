// use ash::vk::{BufferUsageFlags, MemoryPropertyFlags};

use vulkan_renderer::{FrameRenderContext, VulkanRenderer};
// use fractal::{Julia, Mandelbrot};
// use ui::UiBackend;
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

// mod fractal;
mod spin_mutex;
mod vulkan_shader;
// mod ui;
mod math;
mod vulkan_bindless;
mod vulkan_buffer;
mod vulkan_image;
mod vulkan_mapped_memory;
mod vulkan_pipeline;
mod vulkan_renderer;

use enum_iterator::{next_cycle, previous_cycle};

// async fn async_computation(vks: std::sync::Arc<std::pin::Pin<Box<VulkanRenderer>>>) {
//     let (_, _, z) = vks.reserve_staging_memory(1024);
//     println!("{:?} Reserved memory @ offset {z}", std::thread::current());
//     std::thread::sleep(std::time::Duration::from_secs(5));
// }

fn main() {
    // use tokio::runtime;
    // let rt = runtime::Runtime::new().expect("Can't init tokio!");

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

    // use std::sync::Arc;
    //
    // let vks = Arc::new(Box::pin(
    //     VulkanRenderer::new(get_window_data(&window), window.inner_size().into())
    //         .expect("Failed to initialize vulkan ..."),
    // ));
    //
    // let mtx = Arc::new(spin_mutex::SpinMutex::new());
    // let mymtx = Arc::clone(&mtx);
    //
    // let vks2 = Arc::clone(&vks);
    // let task = rt.spawn(async move {
    //     let (x, y, z) = vks2.reserve_staging_memory(1024);
    //     let _ = mtx.lock();
    //     println!("Spwning and waiting for pepega task, reserved mem @ {z}");
    //     let task2 = async_computation(vks2).await;
    //     println!("Original task done.");
    // });
    // {
    //     let _ = mymtx.lock();
    // }
    //
    // rt.block_on(task).unwrap();
    //
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
    // julia: Julia,
    // mandelbrot: Mandelbrot,
    ui_opened: bool,
    // ui: UiBackend,
    control_down: bool,
    cursor_pos: (f32, f32),
    vks: std::pin::Pin<Box<VulkanRenderer>>,
    // bindless_sys: BindlessResourceSystem,
    // ubo_globals_handle: BindlessUniformBufferResourceHandleEntryPair,
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

#[derive(Copy, Clone, Debug)]
struct UniformGlobals {
    world_view_proj: [f32; 16],
    projection: [f32; 16],
    inv_projection: [f32; 16],
    view: [f32; 16],
    ortho_proj: [f32; 16],
    eye_pos: [f32; 4],
    frame_id: u32,
}

impl std::default::Default for UniformGlobals {
    fn default() -> Self {
        unsafe { std::mem::MaybeUninit::<Self>::zeroed().assume_init() }
    }
}

impl FractalSimulation {
    fn new(window: &winit::window::Window) -> FractalSimulation {
        let cursor_pos = (
            (window.inner_size().width / 2) as f32,
            (window.inner_size().height / 2) as f32,
        );

        log::info!("Cursor initial position {:?}", cursor_pos);

        let vks = Box::pin(
            VulkanRenderer::create(get_window_data(window))
                .expect("Failed to initialize vulkan ..."),
        );
        //
        // log::info!("### Device limits:\n{:?}", vks.limits());
        // log::info!("### Device features:\n{:?}", vks.features());
        //
        // let (max_frames,) = vks.setup();
        //
        // let mut bindless_sys = BindlessResourceSystem::new(&vks).expect("Cant create bindless sys");
        //
        // let ui = UiBackend::new(window, &mut vks, &mut bindless_sys, ui::HiDpiMode::Default)
        //     .expect("Failed to create UI backend");
        //
        // let mandelbrot = Mandelbrot::new(&mut vks, &mut bindless_sys);
        // let julia = Julia::new(&mut vks, &mut bindless_sys);
        //
        // let ubo_globals = VulkanBuffer::create(
        //     &mut vks,
        //     &VulkanBufferCreateInfo {
        //         name_tag: None,
        //         work_package: None,
        //         usage: BufferUsageFlags::UNIFORM_BUFFER,
        //         memory_properties: MemoryPropertyFlags::HOST_VISIBLE,
        //         bytes: std::mem::size_of::<UniformGlobals>(),
        //         slabs: max_frames as usize,
        //         initial_data: &[],
        //     },
        // )
        // .expect("Failed to create ubo_globals");
        //
        // let ubo_globals_handle = bindless_sys.register_uniform_buffer(ubo_globals, None);
        //
        FractalSimulation {
            ftype: FractalType::Mandelbrot,
            // julia,
            // mandelbrot,
            ui_opened: true,
            control_down: false,
            cursor_pos,
            vks,
            // ui,
            // bindless_sys,
            // ubo_globals_handle,
        }
    }

    fn begin_rendering(&mut self, window: &winit::window::Window) -> FrameRenderContext {
        let img_size = ash::vk::Extent2D {
            width: window.inner_size().width,
            height: window.inner_size().height,
        };

        self.vks
            .debug_queue_begin_label("### render start ###", [0f32, 1f32, 0f32, 1f32]);
        let frame_context = self.vks.begin_rendering(img_size);

        // UniqueBufferMapping::map_memory(
        //     self.vks.logical(),
        //     self.ubo_globals_handle.1.devmem,
        //     self.ubo_globals_handle.1.aligned_slab_size * frame_context.current_frame_id as usize,
        //     self.ubo_globals_handle.1.aligned_slab_size,
        // )
        // .map(|bmapping| bmapping.write_data(std::slice::from_ref(&ubo_data)));

        // self.bindless_sys.flush_pending_updates(&self.vks);
        // self.bindless_sys
        //     .bind_descriptors(frame_context.cmd_buff, &self.vks);

        frame_context
    }

    // fn setup_ui(&mut self, window: &winit::window::Window) {
    //     let ui = self.ui.new_frame(window);
    //
    //     ui.window("Fractal type")
    //         .size([800f32, 600f32], imgui::Condition::Always)
    //         .resizable(true)
    //         .build(|| {
    //             ui.begin_combo("Fractal type", format!("{:?}", self.ftype))
    //                 .map(|_| {
    //                     let mut selected = self.ftype;
    //                     for item in enum_iterator::all::<FractalType>() {
    //                         if selected == item {
    //                             ui.set_item_default_focus();
    //                         }
    //
    //                         let clicked = ui
    //                             .selectable_config(format!("{:?}", item))
    //                             .selected(selected == item)
    //                             .build();
    //
    //                         //
    //                         // When item is clicked, store it
    //                         if clicked {
    //                             selected = item;
    //                             self.ftype = item;
    //                         }
    //                     }
    //                 });
    //
    //             ui.separator();
    //             ui.text_colored([0.0f32, 1.0f32, 0.0f32, 1.0f32], "::: Parameters :::");
    //             ui.separator();
    //
    //             match self.ftype {
    //                 FractalType::Julia => self.julia.do_ui(ui),
    //                 FractalType::Mandelbrot => self.mandelbrot.do_ui(ui),
    //             }
    //         });
    // }

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
                // self.julia.input_handler(&InputState {
                //     window,
                //     event,
                //     cursor_pos: self.cursor_pos,
                //     control_down: self.control_down,
                // });
                // self.mandelbrot.input_handler(&InputState {
                //     window,
                //     event,
                //     cursor_pos: self.cursor_pos,
                //     control_down: self.control_down,
                // });
            }

            // _ => match self.ftype {
            //     FractalType::Julia => {
            //         self.julia.input_handler(&InputState {
            //             window,
            //             event,
            //             cursor_pos: self.cursor_pos,
            //             control_down: self.control_down,
            //         });
            //     }
            //     FractalType::Mandelbrot => {
            //         self.mandelbrot.input_handler(&InputState {
            //             window,
            //             event,
            //             cursor_pos: self.cursor_pos,
            //             control_down: self.control_down,
            //         });
            //     }
            // },
            _ => {}
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
                // let wants_input = self.ui.handle_event(window, &event);
                // if !wants_input {
                self.handle_window_event(window, win_event, control_flow);
                // }
            }

            Event::MainEventsCleared => {
                let frame_context = self.begin_rendering(window);
                //
                // match self.ftype {
                //     FractalType::Julia => {
                //         self.julia.render(&self.vks, &frame_context);
                //     }
                //     FractalType::Mandelbrot => {
                //         self.mandelbrot.render(&self.vks, &frame_context);
                //     }
                // }
                //
                // self.setup_ui(window);
                // self.ui.draw_frame(&self.vks, &frame_context);
                //
                self.vks.end_rendering(frame_context);
                self.vks.debug_queue_end_label();
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
