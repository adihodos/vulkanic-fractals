use std::{cmp::Ordering, ffi::CStr, mem::size_of};

use ash::vk::{
    BlendFactor, BlendOp, BorderColor, BufferUsageFlags, ColorComponentFlags, CullModeFlags,
    DeviceSize, DynamicState, Extent2D, Filter, FrontFace, ImageType, ImageUsageFlags,
    MemoryPropertyFlags, Offset2D, PipelineBindPoint, PipelineColorBlendAttachmentState,
    PipelineColorBlendStateCreateInfo, PipelineDepthStencilStateCreateInfo, PipelineLayout,
    PipelineRasterizationStateCreateInfo, PolygonMode, Rect2D, SamplerAddressMode,
    SamplerMipmapMode, ShaderStageFlags, Viewport,
};
use imgui::{self, BackendFlags, FontConfig, FontId, Io, Key};
use imgui::{DrawCmd, FontSource};

use winit::{
    dpi::{LogicalPosition, LogicalSize},
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, MouseScrollDelta, TouchPhase,
        VirtualKeyCode, WindowEvent,
    },
    window::{CursorIcon as MouseCursor, Window},
};

use crate::vulkan_bindless::{
    BindlessImageResourceHandleEntryPair, BindlessResourceHandleCore, BindlessResourceSystem,
    BindlessStorageBufferResourceHandleEntryPair, GlobalPushConstant,
};
use crate::vulkan_buffer::{VulkanBuffer, VulkanBufferCreateInfo};
use crate::vulkan_image::{UniqueImage, UniqueSampler, VulkanImageCreateInfo};
use crate::vulkan_pipeline::{
    GraphicsPipelineCreateOptions, GraphicsPipelineSetupHelper, UniquePipeline,
};
use crate::vulkan_renderer::{GraphicsError, QueueType};
use crate::vulkan_shader::ShaderSource;
use crate::{FrameRenderContext, UniqueBufferMapping, VulkanRenderer};

type UiVertex = imgui::DrawVert;
type UiIndex = imgui::DrawIdx;

/// Parts adapted from imgui-winit-support example

/// winit backend platform state
#[derive(Debug)]
pub struct WinitPlatform {
    hidpi_mode: ActiveHiDpiMode,
    hidpi_factor: f64,
    _cursor_cache: Option<CursorSettings>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct CursorSettings {
    cursor: Option<imgui::MouseCursor>,
    draw_cursor: bool,
}

fn to_winit_cursor(cursor: imgui::MouseCursor) -> MouseCursor {
    match cursor {
        imgui::MouseCursor::Arrow => MouseCursor::Default,
        imgui::MouseCursor::TextInput => MouseCursor::Text,
        imgui::MouseCursor::ResizeAll => MouseCursor::Move,
        imgui::MouseCursor::ResizeNS => MouseCursor::NsResize,
        imgui::MouseCursor::ResizeEW => MouseCursor::EwResize,
        imgui::MouseCursor::ResizeNESW => MouseCursor::NeswResize,
        imgui::MouseCursor::ResizeNWSE => MouseCursor::NwseResize,
        imgui::MouseCursor::Hand => MouseCursor::Hand,
        imgui::MouseCursor::NotAllowed => MouseCursor::NotAllowed,
    }
}

impl CursorSettings {
    fn apply(&self, window: &Window) {
        match self.cursor {
            Some(mouse_cursor) if !self.draw_cursor => {
                window.set_cursor_visible(true);
                window.set_cursor_icon(to_winit_cursor(mouse_cursor));
            }
            _ => window.set_cursor_visible(false),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum ActiveHiDpiMode {
    Default,
    Rounded,
    Locked,
}

/// DPI factor handling mode.
///
/// Applications that use imgui-rs might want to customize the used DPI factor and not use
/// directly the value coming from winit.
///
/// **Note: if you use a mode other than default and the DPI factor is adjusted, winit and imgui-rs
/// will use different logical coordinates, so be careful if you pass around logical size or
/// position values.**
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum HiDpiMode {
    /// The DPI factor from winit is used directly without adjustment
    Default,
    /// The DPI factor from winit is rounded to an integer value.
    ///
    /// This prevents the user interface from becoming blurry with non-integer scaling.
    Rounded,
    /// The DPI factor from winit is ignored, and the included value is used instead.
    ///
    /// This is useful if you want to force some DPI factor (e.g. 1.0) and not care about the value
    /// coming from winit.
    Locked(f64),
}

impl HiDpiMode {
    fn apply(&self, hidpi_factor: f64) -> (ActiveHiDpiMode, f64) {
        match *self {
            HiDpiMode::Default => (ActiveHiDpiMode::Default, hidpi_factor),
            HiDpiMode::Rounded => (ActiveHiDpiMode::Rounded, hidpi_factor.round()),
            HiDpiMode::Locked(value) => (ActiveHiDpiMode::Locked, value),
        }
    }
}

fn to_imgui_mouse_button(button: MouseButton) -> Option<imgui::MouseButton> {
    match button {
        MouseButton::Left | MouseButton::Other(0) => Some(imgui::MouseButton::Left),
        MouseButton::Right | MouseButton::Other(1) => Some(imgui::MouseButton::Right),
        MouseButton::Middle | MouseButton::Other(2) => Some(imgui::MouseButton::Middle),
        MouseButton::Other(3) => Some(imgui::MouseButton::Extra1),
        MouseButton::Other(4) => Some(imgui::MouseButton::Extra2),
        _ => None,
    }
}

fn to_imgui_key(keycode: VirtualKeyCode) -> Option<Key> {
    match keycode {
        VirtualKeyCode::Tab => Some(Key::Tab),
        VirtualKeyCode::Left => Some(Key::LeftArrow),
        VirtualKeyCode::Right => Some(Key::RightArrow),
        VirtualKeyCode::Up => Some(Key::UpArrow),
        VirtualKeyCode::Down => Some(Key::DownArrow),
        VirtualKeyCode::PageUp => Some(Key::PageUp),
        VirtualKeyCode::PageDown => Some(Key::PageDown),
        VirtualKeyCode::Home => Some(Key::Home),
        VirtualKeyCode::End => Some(Key::End),
        VirtualKeyCode::Insert => Some(Key::Insert),
        VirtualKeyCode::Delete => Some(Key::Delete),
        VirtualKeyCode::Back => Some(Key::Backspace),
        VirtualKeyCode::Space => Some(Key::Space),
        VirtualKeyCode::Return => Some(Key::Enter),
        VirtualKeyCode::Escape => Some(Key::Escape),
        VirtualKeyCode::LControl => Some(Key::LeftCtrl),
        VirtualKeyCode::LShift => Some(Key::LeftShift),
        VirtualKeyCode::LAlt => Some(Key::LeftAlt),
        VirtualKeyCode::LWin => Some(Key::LeftSuper),
        VirtualKeyCode::RControl => Some(Key::RightCtrl),
        VirtualKeyCode::RShift => Some(Key::RightShift),
        VirtualKeyCode::RAlt => Some(Key::RightAlt),
        VirtualKeyCode::RWin => Some(Key::RightSuper),
        //VirtualKeyCode::Menu => Some(Key::Menu), // TODO: find out if there is a Menu key in winit
        VirtualKeyCode::Key0 => Some(Key::Alpha0),
        VirtualKeyCode::Key1 => Some(Key::Alpha1),
        VirtualKeyCode::Key2 => Some(Key::Alpha2),
        VirtualKeyCode::Key3 => Some(Key::Alpha3),
        VirtualKeyCode::Key4 => Some(Key::Alpha4),
        VirtualKeyCode::Key5 => Some(Key::Alpha5),
        VirtualKeyCode::Key6 => Some(Key::Alpha6),
        VirtualKeyCode::Key7 => Some(Key::Alpha7),
        VirtualKeyCode::Key8 => Some(Key::Alpha8),
        VirtualKeyCode::Key9 => Some(Key::Alpha9),
        VirtualKeyCode::A => Some(Key::A),
        VirtualKeyCode::B => Some(Key::B),
        VirtualKeyCode::C => Some(Key::C),
        VirtualKeyCode::D => Some(Key::D),
        VirtualKeyCode::E => Some(Key::E),
        VirtualKeyCode::F => Some(Key::F),
        VirtualKeyCode::G => Some(Key::G),
        VirtualKeyCode::H => Some(Key::H),
        VirtualKeyCode::I => Some(Key::I),
        VirtualKeyCode::J => Some(Key::J),
        VirtualKeyCode::K => Some(Key::K),
        VirtualKeyCode::L => Some(Key::L),
        VirtualKeyCode::M => Some(Key::M),
        VirtualKeyCode::N => Some(Key::N),
        VirtualKeyCode::O => Some(Key::O),
        VirtualKeyCode::P => Some(Key::P),
        VirtualKeyCode::Q => Some(Key::Q),
        VirtualKeyCode::R => Some(Key::R),
        VirtualKeyCode::S => Some(Key::S),
        VirtualKeyCode::T => Some(Key::T),
        VirtualKeyCode::U => Some(Key::U),
        VirtualKeyCode::V => Some(Key::V),
        VirtualKeyCode::W => Some(Key::W),
        VirtualKeyCode::X => Some(Key::X),
        VirtualKeyCode::Y => Some(Key::Y),
        VirtualKeyCode::Z => Some(Key::Z),
        VirtualKeyCode::F1 => Some(Key::F1),
        VirtualKeyCode::F2 => Some(Key::F2),
        VirtualKeyCode::F3 => Some(Key::F3),
        VirtualKeyCode::F4 => Some(Key::F4),
        VirtualKeyCode::F5 => Some(Key::F5),
        VirtualKeyCode::F6 => Some(Key::F6),
        VirtualKeyCode::F7 => Some(Key::F7),
        VirtualKeyCode::F8 => Some(Key::F8),
        VirtualKeyCode::F9 => Some(Key::F9),
        VirtualKeyCode::F10 => Some(Key::F10),
        VirtualKeyCode::F11 => Some(Key::F11),
        VirtualKeyCode::F12 => Some(Key::F12),
        VirtualKeyCode::Apostrophe => Some(Key::Apostrophe),
        VirtualKeyCode::Comma => Some(Key::Comma),
        VirtualKeyCode::Minus => Some(Key::Minus),
        VirtualKeyCode::Period => Some(Key::Period),
        VirtualKeyCode::Slash => Some(Key::Slash),
        VirtualKeyCode::Semicolon => Some(Key::Semicolon),
        VirtualKeyCode::Equals => Some(Key::Equal),
        VirtualKeyCode::LBracket => Some(Key::LeftBracket),
        VirtualKeyCode::Backslash => Some(Key::Backslash),
        VirtualKeyCode::RBracket => Some(Key::RightBracket),
        VirtualKeyCode::Grave => Some(Key::GraveAccent),
        VirtualKeyCode::Capital => Some(Key::CapsLock),
        VirtualKeyCode::Scroll => Some(Key::ScrollLock),
        VirtualKeyCode::Numlock => Some(Key::NumLock),
        VirtualKeyCode::Snapshot => Some(Key::PrintScreen),
        VirtualKeyCode::Pause => Some(Key::Pause),
        VirtualKeyCode::Numpad0 => Some(Key::Keypad0),
        VirtualKeyCode::Numpad1 => Some(Key::Keypad1),
        VirtualKeyCode::Numpad2 => Some(Key::Keypad2),
        VirtualKeyCode::Numpad3 => Some(Key::Keypad3),
        VirtualKeyCode::Numpad4 => Some(Key::Keypad4),
        VirtualKeyCode::Numpad5 => Some(Key::Keypad5),
        VirtualKeyCode::Numpad6 => Some(Key::Keypad6),
        VirtualKeyCode::Numpad7 => Some(Key::Keypad7),
        VirtualKeyCode::Numpad8 => Some(Key::Keypad8),
        VirtualKeyCode::Numpad9 => Some(Key::Keypad9),
        VirtualKeyCode::NumpadDecimal => Some(Key::KeypadDecimal),
        VirtualKeyCode::NumpadDivide => Some(Key::KeypadDivide),
        VirtualKeyCode::NumpadMultiply => Some(Key::KeypadMultiply),
        VirtualKeyCode::NumpadSubtract => Some(Key::KeypadSubtract),
        VirtualKeyCode::NumpadAdd => Some(Key::KeypadAdd),
        VirtualKeyCode::NumpadEnter => Some(Key::KeypadEnter),
        VirtualKeyCode::NumpadEquals => Some(Key::KeypadEqual),
        _ => None,
    }
}

#[derive(Copy, Clone, Debug)]
// #[repr(C, align(16))]
#[repr(C)]
struct UiBackendParams {
    transform: [f32; 16],
    font_atlas_id: u32,
    _pad: [u32; 3],
}

struct UiRenderState {
    vertex_buffer: VulkanBuffer,
    index_buffer: VulkanBuffer,
    ubo_handle: BindlessStorageBufferResourceHandleEntryPair,
    atlas_handle: BindlessImageResourceHandleEntryPair,
    sampler: UniqueSampler,
    pipeline: UniquePipeline,
    bindless_layout: PipelineLayout,
}

pub struct UiBackend {
    platform: WinitPlatform,
    rs: UiRenderState,
    fontids: Vec<(FontId, String)>,
    imgui: imgui::Context,
}

impl UiBackend {
    const MAX_VERTICES: u32 = 8192;
    const MAX_INDICES: u32 = 16535;
}

impl UiBackend {
    /// Scales a logical size coming from winit using the current DPI mode.
    ///
    /// This utility function is useful if you are using a DPI mode other than default, and want
    /// your application to use the same logical coordinates as imgui-rs.
    pub fn scale_size_from_winit(
        platform: &WinitPlatform,
        window: &Window,
        logical_size: LogicalSize<f64>,
    ) -> LogicalSize<f64> {
        match platform.hidpi_mode {
            ActiveHiDpiMode::Default => logical_size,
            _ => logical_size
                .to_physical::<f64>(window.scale_factor())
                .to_logical(platform.hidpi_factor),
        }
    }

    /// Scales a logical position coming from winit using the current DPI mode.
    ///
    /// This utility function is useful if you are using a DPI mode other than default, and want
    /// your application to use the same logical coordinates as imgui-rs.
    pub fn scale_pos_from_winit(
        platform: &WinitPlatform,
        window: &Window,
        logical_pos: LogicalPosition<f64>,
    ) -> LogicalPosition<f64> {
        match platform.hidpi_mode {
            ActiveHiDpiMode::Default => logical_pos,
            _ => logical_pos
                .to_physical::<f64>(window.scale_factor())
                .to_logical(platform.hidpi_factor),
        }
    }

    /// Scales a logical position for winit using the current DPI mode.
    ///
    /// This utility function is useful if you are using a DPI mode other than default, and want
    /// your application to use the same logical coordinates as imgui-rs.
    pub fn scale_pos_for_winit(
        platform: &WinitPlatform,
        window: &Window,
        logical_pos: LogicalPosition<f64>,
    ) -> LogicalPosition<f64> {
        match platform.hidpi_mode {
            ActiveHiDpiMode::Default => logical_pos,
            _ => logical_pos
                .to_physical::<f64>(platform.hidpi_factor)
                .to_logical(window.scale_factor()),
        }
    }

    pub fn new(
        window: &winit::window::Window,
        vks: &mut VulkanRenderer,
        bindless_sys: &mut BindlessResourceSystem,
        hidpi_mode: HiDpiMode,
    ) -> Result<UiBackend, GraphicsError> {
        let mut imgui = init_imgui();

        let (hidpi_mode, hidpi_factor) = hidpi_mode.apply(window.scale_factor());

        let platform = WinitPlatform {
            hidpi_mode,
            hidpi_factor,
            _cursor_cache: None,
        };

        imgui.io_mut().display_framebuffer_scale = [hidpi_factor as f32, hidpi_factor as f32];
        let logical_size = window.inner_size().to_logical(hidpi_factor);
        let logical_size = Self::scale_size_from_winit(&platform, window, logical_size);
        imgui.io_mut().display_size = [logical_size.width as f32, logical_size.height as f32];
        imgui.io_mut().mouse_pos = [0f32; 2];

        let vertex_buffer = VulkanBuffer::create(
            vks,
            &VulkanBufferCreateInfo {
                name_tag: Some("[[UI]] vertex buffer"),
                work_package: None,
                usage: BufferUsageFlags::VERTEX_BUFFER,
                memory_properties: MemoryPropertyFlags::HOST_VISIBLE,
                slabs: vks.max_frames() as usize,
                bytes: Self::MAX_VERTICES as usize * size_of::<UiVertex>(),
                initial_data: &[],
            },
        )?;

        let index_buffer = VulkanBuffer::create(
            vks,
            &VulkanBufferCreateInfo {
                name_tag: Some("[[UI]] index buffer"),
                work_package: None,
                usage: BufferUsageFlags::INDEX_BUFFER,
                memory_properties: MemoryPropertyFlags::HOST_VISIBLE,
                slabs: vks.max_frames() as usize,
                bytes: Self::MAX_INDICES as usize * size_of::<UiIndex>(),
                initial_data: &[],
            },
        )?;

        let ubo_vs = VulkanBuffer::create(
            vks,
            &VulkanBufferCreateInfo {
                name_tag: Some("[[UI]] Backend UBO"),
                work_package: None,
                usage: BufferUsageFlags::STORAGE_BUFFER,
                memory_properties: MemoryPropertyFlags::HOST_VISIBLE,
                slabs: vks.max_frames() as usize,
                bytes: std::mem::size_of::<UiBackendParams>(),
                initial_data: &[],
            },
        )?;

        let ubo_handle = bindless_sys.register_storage_buffer(ubo_vs, None);
        log::info!("ui ubo {ubo_handle:?}");

        let fontids = [
            ("data/fonts/iosevka-ss03-regular.ttf", "IosevkaRegular"),
            ("data/fonts/iosevka-ss03-medium.ttf", "IosevkaMedium"),
            ("data/fonts/RobotoMono-Medium.ttf", "RobotoMonoMedium"),
            ("data/fonts/RobotoMono-Regular.ttf", "RobotoMonoRegular"),
        ]
        .iter()
        .filter_map(|(font_file, font_name)| {
            std::fs::File::open(font_file)
                .and_then(|mut ff| {
                    let mut ttf_bytes = Vec::<u8>::new();
                    use std::io::Read;
                    ff.read_to_end(&mut ttf_bytes)
                        .map(|_| (ttf_bytes, font_name))
                })
                .and_then(|(font_bytes, font_name)| {
                    Ok((
                        imgui.fonts().add_font(&[FontSource::TtfData {
                            data: &font_bytes,
                            size_pixels: 18f32,
                            config: Some(FontConfig {
                                oversample_h: 4,
                                oversample_v: 4,
                                rasterizer_multiply: 1.5f32,
                                ..FontConfig::default()
                            }),
                        }]),
                        font_name.to_string(),
                    ))
                })
                .ok()
        })
        .collect::<Vec<(FontId, String)>>();

        let baked_font_atlas_image = imgui.fonts().build_alpha8_texture();
        let qjob = vks.create_queue_job(QueueType::Transfer)?;
        let font_atlas_img = UniqueImage::from_bytes(
            vks,
            &VulkanImageCreateInfo {
                tag_name: Some("UI font atlas"),
                work_pkg: Some(&qjob),
                ty: ImageType::TYPE_2D,
                usage: ImageUsageFlags::SAMPLED,
                memory: MemoryPropertyFlags::DEVICE_LOCAL,
                format: ash::vk::Format::R8_UNORM,
                cubemap: false,
                width: baked_font_atlas_image.width,
                height: baked_font_atlas_image.height,
                depth: 1,
                layers: 1,
                pixels: &[baked_font_atlas_image.data],
            },
        )?;

        let wait_token = vks.submit_queue_job(qjob)?;

        let sampler = UniqueSampler::new(
            vks,
            ash::vk::SamplerCreateInfo::default()
                .address_mode_u(SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(SamplerAddressMode::CLAMP_TO_EDGE)
                .border_color(BorderColor::INT_OPAQUE_BLACK)
                .mag_filter(Filter::LINEAR)
                .min_filter(Filter::LINEAR)
                .mipmap_mode(SamplerMipmapMode::LINEAR),
        )?;

        let atlas_handle = bindless_sys.register_image(font_atlas_img, &sampler, None);
        vks.queue_ownership_transfer(&atlas_handle);

        let pipeline = GraphicsPipelineSetupHelper::new()
            .set_input_assembly_state(crate::vulkan_pipeline::InputAssemblyState {
                stride: std::mem::size_of::<UiVertex>() as u32,
                input_rate: ash::vk::VertexInputRate::VERTEX,
                vertex_descriptions: vec![
                    ash::vk::VertexInputAttributeDescription::default()
                        .binding(0)
                        .format(ash::vk::Format::R32G32_SFLOAT)
                        .location(0)
                        .offset(0),
                    ash::vk::VertexInputAttributeDescription::default()
                        .binding(0)
                        .format(ash::vk::Format::R32G32_SFLOAT)
                        .location(1)
                        .offset(8),
                    ash::vk::VertexInputAttributeDescription::default()
                        .binding(0)
                        .format(ash::vk::Format::R8G8B8A8_UNORM)
                        .location(2)
                        .offset(16),
                ],
            })
            .add_shader_stage(ShaderSource::File("data/shaders/ui.bindless.vert".into()))
            .add_shader_stage(ShaderSource::File("data/shaders/ui.bindless.frag".into()))
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
            .set_colorblend_state(
                PipelineColorBlendStateCreateInfo::default().attachments(&[
                    PipelineColorBlendAttachmentState::default()
                        .color_write_mask(ColorComponentFlags::RGBA)
                        .blend_enable(true)
                        .alpha_blend_op(BlendOp::ADD)
                        .color_blend_op(BlendOp::ADD)
                        .src_color_blend_factor(BlendFactor::SRC_ALPHA)
                        .dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
                        .src_alpha_blend_factor(BlendFactor::ONE)
                        .dst_alpha_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA),
                ]),
            )
            .set_dynamic_state(&[
                DynamicState::VIEWPORT_WITH_COUNT,
                DynamicState::SCISSOR_WITH_COUNT,
            ])
            .create(
                vks,
                GraphicsPipelineCreateOptions {
                    layout: Some(bindless_sys.pipeline_layout()),
                },
            )?;

        vks.consume_wait_token(wait_token);

        Ok(UiBackend {
            imgui,

            rs: UiRenderState {
                vertex_buffer,
                index_buffer,
                ubo_handle,
                atlas_handle,
                sampler,
                pipeline,
                bindless_layout: bindless_sys.pipeline_layout(),
            },

            fontids,
            platform,
        })
    }

    fn handle_key_modifier(io: &mut Io, key: VirtualKeyCode, down: bool) {
        if key == VirtualKeyCode::LShift || key == VirtualKeyCode::RShift {
            io.add_key_event(imgui::Key::ModShift, down);
        } else if key == VirtualKeyCode::LControl || key == VirtualKeyCode::RControl {
            io.add_key_event(imgui::Key::ModCtrl, down);
        } else if key == VirtualKeyCode::LAlt || key == VirtualKeyCode::RAlt {
            io.add_key_event(imgui::Key::ModAlt, down);
        } else if key == VirtualKeyCode::LWin || key == VirtualKeyCode::RWin {
            io.add_key_event(imgui::Key::ModSuper, down);
        }
    }

    /// Handles a winit event.
    ///
    /// This function performs the following actions (depends on the event):
    ///
    /// * window size / dpi factor changes are applied
    /// * keyboard state is updated
    /// * mouse state is updated
    pub fn handle_event<T>(&mut self, window: &Window, event: &Event<T>) -> bool {
        match *event {
            Event::WindowEvent {
                window_id,
                ref event,
            } if window_id == window.id() => {
                self.handle_window_event(window, event);
            }
            // Track key release events outside our window. If we don't do this,
            // we might never see the release event if some other window gets focus.
            Event::DeviceEvent {
                event:
                    DeviceEvent::Key(KeyboardInput {
                        state: ElementState::Released,
                        virtual_keycode: Some(key),
                        ..
                    }),
                ..
            } => {
                if let Some(key) = to_imgui_key(key) {
                    self.imgui.io_mut().add_key_event(key, false);
                }
            }
            _ => (),
        }

        let (wants_keys, wants_mouse) = (
            self.imgui.io().want_capture_keyboard,
            self.imgui.io().want_capture_mouse,
        );

        wants_keys || wants_mouse
    }

    fn handle_window_event(&mut self, window: &Window, event: &WindowEvent) {
        match *event {
            WindowEvent::Resized(physical_size) => {
                let logical_size = physical_size.to_logical(window.scale_factor());
                let logical_size =
                    Self::scale_size_from_winit(&self.platform, window, logical_size);
                self.imgui.io_mut().display_size =
                    [logical_size.width as f32, logical_size.height as f32];
            }

            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                let hidpi_factor = match self.platform.hidpi_mode {
                    ActiveHiDpiMode::Default => scale_factor,
                    ActiveHiDpiMode::Rounded => scale_factor.round(),
                    _ => return,
                };
                // Mouse position needs to be changed while we still have both the old and the new
                // values
                if self.imgui.io().mouse_pos[0].is_finite()
                    && self.imgui.io().mouse_pos[1].is_finite()
                {
                    self.imgui.io_mut().mouse_pos = [
                        self.imgui.io().mouse_pos[0]
                            * (hidpi_factor / self.platform.hidpi_factor) as f32,
                        self.imgui.io().mouse_pos[1]
                            * (hidpi_factor / self.platform.hidpi_factor) as f32,
                    ];
                }

                self.platform.hidpi_factor = hidpi_factor;
                self.imgui.io_mut().display_framebuffer_scale =
                    [hidpi_factor as f32, hidpi_factor as f32];
                // Window size might change too if we are using DPI rounding
                let logical_size = window.inner_size().to_logical(scale_factor);
                let logical_size =
                    Self::scale_size_from_winit(&self.platform, window, logical_size);
                self.imgui.io_mut().display_size =
                    [logical_size.width as f32, logical_size.height as f32];
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                // We need to track modifiers separately because some system like macOS, will
                // not reliably send modifier states during certain events like ScreenCapture.
                // Gotta let the people show off their pretty imgui widgets!
                let io = self.imgui.io_mut();
                io.add_key_event(Key::ModShift, modifiers.shift());
                io.add_key_event(Key::ModCtrl, modifiers.ctrl());
                io.add_key_event(Key::ModAlt, modifiers.alt());
                io.add_key_event(Key::ModSuper, modifiers.logo());
            }
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(key),
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state == ElementState::Pressed;

                // We map both left and right ctrl to `ModCtrl`, etc.
                // imgui is told both "left control is pressed" and
                // "consider the control key is pressed". Allows
                // applications to use either general "ctrl" or a
                // specific key. Same applies to other modifiers.
                // https://github.com/ocornut/imgui/issues/5047
                Self::handle_key_modifier(self.imgui.io_mut(), key, pressed);

                // Add main key event
                if let Some(key) = to_imgui_key(key) {
                    self.imgui.io_mut().add_key_event(key, pressed);
                }
            }
            WindowEvent::ReceivedCharacter(ch) => {
                // Exclude the backspace key ('\u{7f}'). Otherwise we will insert this char and then
                // delete it.
                if ch != '\u{7f}' {
                    self.imgui.io_mut().add_input_character(ch);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let position = position.to_logical(window.scale_factor());
                let position = Self::scale_pos_from_winit(&self.platform, window, position);
                self.imgui
                    .io_mut()
                    .add_mouse_pos_event([position.x as f32, position.y as f32]);
            }
            WindowEvent::MouseWheel {
                delta,
                phase: TouchPhase::Moved,
                ..
            } => {
                let (h, v) = match delta {
                    MouseScrollDelta::LineDelta(h, v) => (h, v),
                    MouseScrollDelta::PixelDelta(pos) => {
                        let pos = pos.to_logical::<f64>(self.platform.hidpi_factor);
                        let h = match pos.x.partial_cmp(&0.0) {
                            Some(Ordering::Greater) => 1.0,
                            Some(Ordering::Less) => -1.0,
                            _ => 0.0,
                        };
                        let v = match pos.y.partial_cmp(&0.0) {
                            Some(Ordering::Greater) => 1.0,
                            Some(Ordering::Less) => -1.0,
                            _ => 0.0,
                        };
                        (h, v)
                    }
                };
                self.imgui.io_mut().add_mouse_wheel_event([h, v]);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(mb) = to_imgui_mouse_button(button) {
                    let pressed = state == ElementState::Pressed;
                    self.imgui.io_mut().add_mouse_button_event(mb, pressed);
                }
            }
            WindowEvent::Focused(newly_focused) => {
                if !newly_focused {
                    // Set focus-lost to avoid stuck keys (like 'alt'
                    // when alt-tabbing)
                    self.imgui.io_mut().app_focus_lost = true;
                }
            }
            _ => (),
        }
    }

    pub fn new_frame(&mut self, window: &winit::window::Window) -> &mut imgui::Ui {
        if self.imgui.io().want_set_mouse_pos {
            let logical_pos = Self::scale_pos_for_winit(
                &self.platform,
                window,
                LogicalPosition::new(
                    f64::from(self.imgui.io().mouse_pos[0]),
                    f64::from(self.imgui.io().mouse_pos[1]),
                ),
            );
            let _ = window.set_cursor_position(logical_pos);
        }
        self.imgui.new_frame()
    }

    pub fn draw_frame(&mut self, vks: &VulkanRenderer, frame_context: &FrameRenderContext) {
        let ui_context = &mut self.imgui;

        let draw_data = ui_context.render();
        assert!(draw_data.total_vtx_count < Self::MAX_VERTICES as i32);
        assert!(draw_data.total_idx_count < Self::MAX_INDICES as i32);

        let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];
        if !(fb_width > 0.0 && fb_height > 0.0) {
            return;
        }

        if draw_data.total_vtx_count < 1 || draw_data.total_idx_count < 1 {
            return;
        }

        //
        // Push vertices + indices 2 GPU
        {
            let mut vertex_buffer_mapping = UniqueBufferMapping::map_buffer(
                &self.rs.vertex_buffer,
                &vks.logical(),
                Some(
                    self.rs.vertex_buffer.aligned_slab_size
                        * frame_context.current_frame_id as usize,
                ),
                Some(self.rs.vertex_buffer.aligned_slab_size),
            )
            .unwrap();

            let mut index_buffer_mapping = UniqueBufferMapping::map_buffer(
                &self.rs.index_buffer,
                &vks.logical(),
                Some(
                    self.rs.index_buffer.aligned_slab_size
                        * frame_context.current_frame_id as usize,
                ),
                Some(self.rs.index_buffer.aligned_slab_size),
            )
            .unwrap();

            let _ = draw_data.draw_lists().fold(
                (0isize, 0isize),
                |(vtx_offset, idx_offset), draw_list| {
                    vertex_buffer_mapping
                        .write_data_with_offset(draw_list.vtx_buffer(), vtx_offset);
                    index_buffer_mapping.write_data_with_offset(draw_list.idx_buffer(), idx_offset);

                    (
                        vtx_offset + draw_list.vtx_buffer().len() as isize,
                        idx_offset + draw_list.idx_buffer().len() as isize,
                    )
                },
            );
        }

        let graphics_device = vks.logical();

        unsafe {
            graphics_device.cmd_bind_pipeline(
                frame_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.rs.pipeline.handle(),
            );

            graphics_device.cmd_bind_vertex_buffers(
                frame_context.cmd_buff,
                0,
                &[self.rs.vertex_buffer.buffer],
                &[self.rs.vertex_buffer.aligned_slab_size as DeviceSize
                    * frame_context.current_frame_id as DeviceSize],
            );

            graphics_device.cmd_bind_index_buffer(
                frame_context.cmd_buff,
                self.rs.index_buffer.buffer,
                self.rs.index_buffer.aligned_slab_size as DeviceSize
                    * frame_context.current_frame_id as DeviceSize,
                ash::vk::IndexType::UINT16,
            );

            graphics_device.cmd_set_viewport_with_count(
                frame_context.cmd_buff,
                &[Viewport {
                    x: 0f32,
                    y: 0f32,
                    width: frame_context.fb_size.width as f32,
                    height: frame_context.fb_size.height as f32,
                    min_depth: 0f32,
                    max_depth: 1f32,
                }],
            );

            let scale = [
                2f32 / draw_data.display_size[0],
                2f32 / draw_data.display_size[1],
            ];

            let translate = [
                -1f32 - draw_data.display_pos[0] * scale[0],
                -1f32 - draw_data.display_pos[1] * scale[1],
            ];

            let atlas_id = self.rs.atlas_handle.0.handle();

            let transform = UiBackendParams {
                transform: [
                    scale[0],
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    scale[1],
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    1.0f32,
                    0.0f32,
                    translate[0],
                    translate[1],
                    0.0f32,
                    1.0f32,
                ],
                font_atlas_id: atlas_id,
                _pad: [0u32; 3],
            };

            //
            // push transform
            UniqueBufferMapping::map_memory(
                vks.logical(),
                self.rs.ubo_handle.1.devmem,
                self.rs.ubo_handle.1.aligned_slab_size * frame_context.current_frame_id as usize,
                self.rs.ubo_handle.1.aligned_slab_size,
            )
            .map(|ubo| ubo.write_data(std::slice::from_ref(&transform)))
            .expect("Failed to update UI params");

            graphics_device.cmd_push_constants(
                frame_context.cmd_buff,
                self.rs.bindless_layout,
                ShaderStageFlags::ALL,
                0,
                &GlobalPushConstant::from_resource(
                    self.rs.ubo_handle.0,
                    frame_context.current_frame_id,
                    None,
                )
                .to_gpu(), // &UIPushConstant::create(self.rs.ubo_handle.0, frame_context.current_frame_id)
                           //     .to_gpu(),
            );

            //
            // Will project scissor/clipping rectangles into framebuffer space
            let clip_off = draw_data.display_pos;
            let clip_scale = draw_data.framebuffer_scale;

            let _ = draw_data.draw_lists().fold(
                (0u32, 0u32),
                |(vertex_offset, index_offset), draw_list| {
                    for draw_cmd in draw_list.commands() {
                        match draw_cmd {
                            DrawCmd::Elements { count, cmd_params } => {
                                let mut clip_min = [
                                    (cmd_params.clip_rect[0] - clip_off[0]) * clip_scale[0],
                                    (cmd_params.clip_rect[1] - clip_off[1]) * clip_scale[1],
                                ];
                                let mut clip_max = [
                                    (cmd_params.clip_rect[2] - clip_off[0]) * clip_scale[0],
                                    (cmd_params.clip_rect[3] - clip_off[1]) * clip_scale[1],
                                ];
                                //
                                // Clamp to viewport as vkCmdSetScissor() won't accept values that are off bounds
                                if clip_min[0] < 0f32 {
                                    clip_min[0] = 0f32;
                                }

                                if clip_min[1] < 0f32 {
                                    clip_min[1] = 0f32;
                                }

                                if clip_max[0] > fb_width as f32 {
                                    clip_max[0] = fb_width as f32;
                                }

                                if clip_max[1] > fb_height as f32 {
                                    clip_max[1] = fb_height as f32;
                                }

                                if clip_max[0] <= clip_min[0] || clip_max[1] <= clip_min[1] {
                                    continue;
                                }

                                let scissor = [Rect2D {
                                    offset: Offset2D {
                                        x: clip_min[0] as i32,
                                        y: clip_min[1] as i32,
                                    },
                                    extent: Extent2D {
                                        width: (clip_max[0] - clip_min[0]).abs() as u32,
                                        height: (clip_max[1] - clip_min[1]).abs() as u32,
                                    },
                                }];

                                graphics_device
                                    .cmd_set_scissor_with_count(frame_context.cmd_buff, &scissor);

                                graphics_device.cmd_draw_indexed(
                                    frame_context.cmd_buff,
                                    count as u32,
                                    1,
                                    index_offset as u32 + cmd_params.idx_offset as u32,
                                    vertex_offset as i32 + cmd_params.vtx_offset as i32,
                                    0,
                                );
                            }
                            DrawCmd::ResetRenderState => log::info!("reset render state"),
                            _ => {}
                        }
                    }

                    (
                        vertex_offset + draw_list.vtx_buffer().len() as u32,
                        index_offset + draw_list.idx_buffer().len() as u32,
                    )
                },
            );
        }
    }
}

fn init_imgui() -> imgui::Context {
    let mut imgui = imgui::Context::create();
    let io = imgui.io_mut();

    io.backend_flags.insert(BackendFlags::HAS_MOUSE_CURSORS);
    io.backend_flags.insert(BackendFlags::HAS_SET_MOUSE_POS);
    imgui.set_platform_name(Some(format!(
        "FractalExplorer {}",
        env!("CARGO_PKG_VERSION")
    )));

    imgui
}

// #[derive(Copy, Clone, Debug)]
// struct UIPushConstant(u32);
//
// impl UIPushConstant {
//     fn create<T>(bindless_res: BindlessResourceHandleCore<T>, frame_id: u32) -> Self {
//         assert!(frame_id < 16);
//         let resource_handle = bindless_res.element_handle(frame_id as usize).handle();
//         Self((resource_handle << 4) | frame_id)
//     }
//
//     fn to_gpu(self) -> [u8; 4] {
//         self.0.to_le_bytes()
//     }
// }
