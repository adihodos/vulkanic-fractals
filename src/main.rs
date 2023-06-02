use std::{
    ffi::{c_char, CString},
    mem::size_of,
    os::raw::c_void,
};

use ash::{
    extensions::ext::DebugUtils,
    vk::{
        AccessFlags, ApplicationInfo, AttachmentDescription, AttachmentLoadOp, AttachmentReference,
        AttachmentStoreOp, Buffer, BufferCreateInfo, BufferImageCopy, BufferUsageFlags,
        ClearColorValue, ClearValue, ColorComponentFlags, CommandBuffer, CommandBufferAllocateInfo,
        CommandBufferBeginInfo, CommandBufferLevel, CommandBufferResetFlags,
        CommandBufferUsageFlags, CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo,
        ComponentMapping, ComponentSwizzle, CompositeAlphaFlagsKHR, CullModeFlags,
        DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
        DebugUtilsMessengerCreateInfoEXT, DebugUtilsMessengerEXT, DependencyFlags,
        DescriptorBufferInfo, DescriptorPool, DescriptorPoolCreateInfo, DescriptorPoolSize,
        DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding,
        DescriptorSetLayoutCreateInfo, DescriptorType, DeviceCreateInfo, DeviceMemory,
        DeviceQueueCreateInfo, DeviceSize, DynamicState, Extent2D, Fence, FenceCreateFlags,
        FenceCreateInfo, Format, Framebuffer, FramebufferCreateInfo, FrontFace,
        GraphicsPipelineCreateInfo, Image, ImageAspectFlags, ImageCreateInfo, ImageLayout,
        ImageMemoryBarrier, ImageSubresource, ImageSubresourceLayers, ImageSubresourceRange,
        ImageUsageFlags, ImageView, ImageViewCreateInfo, ImageViewType, InstanceCreateInfo,
        MappedMemoryRange, MemoryAllocateInfo, MemoryMapFlags, MemoryPropertyFlags,
        MemoryRequirements, Offset2D, PhysicalDevice, PhysicalDeviceFeatures,
        PhysicalDeviceMemoryProperties, PhysicalDeviceProperties, PhysicalDeviceType, Pipeline,
        PipelineBindPoint, PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateInfo,
        PipelineDynamicStateCreateInfo, PipelineInputAssemblyStateCreateInfo, PipelineLayout,
        PipelineLayoutCreateInfo, PipelineMultisampleStateCreateInfo,
        PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags,
        PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo, PolygonMode,
        PresentInfoKHR, PresentModeKHR, PrimitiveTopology, Queue, QueueFlags, Rect2D, RenderPass,
        RenderPassBeginInfo, RenderPassCreateInfo, SampleCountFlags, Semaphore,
        SemaphoreCreateInfo, ShaderModule, ShaderModuleCreateInfo, ShaderStageFlags, SharingMode,
        SubmitInfo, SubpassContents, SubpassDescription, SurfaceFormatKHR, SurfaceKHR,
        SurfaceTransformFlagsKHR, SwapchainCreateInfoKHR, SwapchainKHR,
        VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, Viewport,
        WriteDescriptorSet, XlibSurfaceCreateInfoKHR, WHOLE_SIZE,
    },
    Device, Entry, Instance,
};

use ui::UiBackend;
use vulkan_renderer::UniqueImage;
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

mod ui;
mod vulkan_renderer;

use enum_iterator::{next_cycle, previous_cycle};

use crate::vulkan_renderer::{choose_memory_heap, compile_shader_from_file};

#[derive(Copy, Clone, Debug)]
pub struct WindowSystemIntegration {
    pub native_disp: *mut std::os::raw::c_void,
    pub native_win: std::os::raw::c_ulong,
}

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

struct FractalSimulation {
    ui_opened: bool,
    ui: UiBackend,
    control_down: bool,
    cursor_pos: (f32, f32),
    fractal: Fractal,
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
        let fractal = Fractal::new(&vks);
        let ui = UiBackend::new(window, &mut vks, ui::HiDpiMode::Default);
        vks.end_resource_loading();

        FractalSimulation {
            ui_opened: true,
            control_down: false,
            cursor_pos,
            fractal,
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
        ui.show_demo_window(&mut self.ui_opened);
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
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => control_flow.set_exit(),

            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos.0 = position.x as f32;
                self.cursor_pos.1 = position.y as f32;
            }

            WindowEvent::Resized(new_size) => {
                self.fractal
                    .params
                    .screen_resized(new_size.width, new_size.height);
            }

            WindowEvent::ModifiersChanged(mods) => {
                self.control_down = mods.ctrl();
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let zoom_in = match delta {
                    winit::event::MouseScrollDelta::LineDelta(.., y) => y > 0f32,
                    winit::event::MouseScrollDelta::PixelDelta(PhysicalPosition { y, .. }) => {
                        y > 0f64
                    }
                };

                if !zoom_in {
                    self.fractal.params.zoom_out();
                } else {
                    self.fractal.params.zoom_in();
                }
            }

            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match virtual_keycode {
                Some(VirtualKeyCode::Back) => {
                    self.fractal.params = FractalParameters {
                        screen_width: window.inner_size().width,
                        screen_height: window.inner_size().height,
                        ..Default::default()
                    };
                }

                Some(VirtualKeyCode::PageUp) => {
                    self.fractal.params.previous_color();
                }

                Some(VirtualKeyCode::PageDown) => {
                    self.fractal.params.next_color();
                }

                Some(VirtualKeyCode::NumpadSubtract) => {
                    self.fractal.params.decrease_iterations();
                }

                Some(VirtualKeyCode::NumpadAdd) => {
                    self.fractal.params.increase_iterations();
                }

                Some(VirtualKeyCode::Insert) => {
                    self.fractal.params.escape_radius = (self.fractal.params.escape_radius * 2)
                        .min(FractalParameters::ESC_RADIUS_MAX);
                }

                Some(VirtualKeyCode::Delete) => {
                    self.fractal.params.escape_radius = (self.fractal.params.escape_radius / 2)
                        .max(FractalParameters::ESC_RADIUS_MIN);
                }

                _ => {}
            },

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => match button {
                MouseButton::Left => {
                    self.fractal.params.center_moved(
                        self.cursor_pos.0,
                        self.cursor_pos.1,
                        self.control_down,
                    );
                }
                MouseButton::Right => {
                    self.fractal.params.zoom_out();
                }
                _ => {}
            },

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
                let wants_input = self.ui.handle_event(window, &event);
                if !wants_input {
                    self.handle_window_event(window, win_event, control_flow);
                }
            }

            Event::MainEventsCleared => {
                let frame_context = self.begin_rendering();

                self.fractal.render(&self.vks, &frame_context);

                self.setup_ui(window);
                self.ui.draw_frame(&self.vks, &frame_context);

                self.end_rendering(&frame_context);
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

struct ResourceLoadingState {
    cmd_buf: CommandBuffer,
    fence: Fence,
    work_buffers: Vec<UniqueBuffer>,
}

impl ResourceLoadingState {
    fn new(ds: &VulkanDeviceState) -> ResourceLoadingState {
        let cmd_buf = unsafe {
            ds.device.allocate_command_buffers(
                &CommandBufferAllocateInfo::builder()
                    .command_buffer_count(1)
                    .command_pool(ds.cmd_pool)
                    .level(CommandBufferLevel::PRIMARY),
            )
        }
        .expect("Failed to allocate command buffer")[0];

        let fence = unsafe { ds.device.create_fence(&FenceCreateInfo::builder(), None) }
            .expect("Failed to create fence");

        ResourceLoadingState {
            cmd_buf,
            fence,
            work_buffers: vec![],
        }
    }
}

pub struct VulkanState {
    resource_loader: ResourceLoadingState,
    pub renderpass: RenderPass,
    pub swapchain: VulkanSwapchainState,
    pub ds: VulkanDeviceState,
    pub msgr: DebugUtilsMessengerEXT,
    pub dbg: DebugUtils,
    pub instance: Instance,
    pub entry: Entry,
}

impl VulkanState {
    pub fn wait_all_idle(&mut self) {
        unsafe {
            self.ds
                .device
                .queue_wait_idle(self.ds.queue)
                .expect("Failed to wait for idle queue");
            self.ds
                .device
                .device_wait_idle()
                .expect("Failed to wait for device idle");
        }
    }

    pub fn begin_resource_loading(&self) {
        unsafe {
            self.ds.device.begin_command_buffer(
                self.resource_loader.cmd_buf,
                &CommandBufferBeginInfo::builder().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .expect("Failed to begin command buffer for resource loading");
    }

    pub fn end_resource_loading(&mut self) {
        unsafe {
            self.ds
                .device
                .end_command_buffer(self.resource_loader.cmd_buf)
                .expect("Failed to end command buffer");
            self.ds
                .device
                .queue_submit(
                    self.ds.queue,
                    &[*SubmitInfo::builder().command_buffers(&[self.resource_loader.cmd_buf])],
                    self.resource_loader.fence,
                )
                .expect("Failed to submit command buffer");
            self.ds
                .device
                .wait_for_fences(&[self.resource_loader.fence], true, u64::MAX)
                .expect("Failed to wait for fences ...");
        }

        self.resource_loader.work_buffers.clear();
    }

    pub fn copy_pixels_to_image(
        &mut self,
        img: &UniqueImage,
        pixels: &[u8],
        image_info: &ImageCreateInfo,
    ) {
        let work_buffer = UniqueBuffer::new::<u8>(
            self,
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_SRC,
            MemoryPropertyFlags::HOST_VISIBLE,
            pixels.len(),
        );

        {
            let mapped_buffer = UniqueBufferMapping::new(&work_buffer, &self.ds, None, None);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    pixels.as_ptr(),
                    mapped_buffer.mapped_memory as *mut u8,
                    pixels.len(),
                );
            }
        }

        let img_subresource_range = *ImageSubresourceRange::builder()
            .aspect_mask(ImageAspectFlags::COLOR)
            .layer_count(image_info.array_layers)
            .base_array_layer(0)
            .level_count(image_info.mip_levels)
            .base_mip_level(0);

        //
        // transition image layout from undefined -> transfer src
        unsafe {
            self.ds.device.cmd_pipeline_barrier(
                self.resource_loader.cmd_buf,
                PipelineStageFlags::TOP_OF_PIPE,
                PipelineStageFlags::TRANSFER,
                DependencyFlags::empty(),
                &[],
                &[],
                &[*ImageMemoryBarrier::builder()
                    .src_access_mask(AccessFlags::NONE)
                    .dst_access_mask(AccessFlags::TRANSFER_WRITE)
                    .image(img.image)
                    .old_layout(ImageLayout::UNDEFINED)
                    .new_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
                    .subresource_range(img_subresource_range)],
            );
        }

        //
        // copy pixels
        unsafe {
            self.ds.device.cmd_copy_buffer_to_image(
                self.resource_loader.cmd_buf,
                work_buffer.handle,
                img.image,
                ImageLayout::TRANSFER_DST_OPTIMAL,
                &[*BufferImageCopy::builder()
                    .buffer_offset(0)
                    .image_subresource(
                        *ImageSubresourceLayers::builder()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .base_array_layer(0)
                            .layer_count(image_info.array_layers)
                            .mip_level(0),
                    )
                    .image_extent(image_info.extent)],
            );
        }

        //
        // transition layout from transfer -> shader readonly optimal
        unsafe {
            self.ds.device.cmd_pipeline_barrier(
                self.resource_loader.cmd_buf,
                PipelineStageFlags::TRANSFER,
                PipelineStageFlags::FRAGMENT_SHADER,
                DependencyFlags::empty(),
                &[],
                &[],
                &[*ImageMemoryBarrier::builder()
                    .src_access_mask(AccessFlags::MEMORY_READ)
                    .dst_access_mask(AccessFlags::MEMORY_WRITE)
                    .image(img.image)
                    .old_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .subresource_range(img_subresource_range)],
            );
        }

        self.resource_loader.work_buffers.push(work_buffer);
    }
}

impl std::ops::Drop for VulkanState {
    fn drop(&mut self) {}
}

pub struct VulkanPhysicalDeviceState {
    pub device: PhysicalDevice,
    pub properties: PhysicalDeviceProperties,
    pub memory_properties: PhysicalDeviceMemoryProperties,
    pub features: PhysicalDeviceFeatures,
    pub queue_family_id: u32,
}

pub struct VulkanDeviceState {
    pub descriptor_pool: DescriptorPool,
    pub cmd_pool: CommandPool,
    pub queue: Queue,
    pub device: ash::Device,
    pub surface: VulkanSurfaceState,
    pub physical: VulkanPhysicalDeviceState,
}

pub struct VulkanSurfaceKHRState {
    pub ext: ash::extensions::khr::Surface,
    pub surface: SurfaceKHR,
}

pub struct VulkanSurfaceState {
    pub khr: VulkanSurfaceKHRState,
    pub fmt: SurfaceFormatKHR,
    pub present_mode: PresentModeKHR,
    pub transform: SurfaceTransformFlagsKHR,
    pub image_count: u32,
    pub image_size: Extent2D,
}

// struct PresentState {
//     cmd_buf: CommandBuffer,
//     fence: Fence,
//     sem_img_avail: Semaphore,
//     sem_img_finished: Semaphore,
// }

pub struct VulkanSwapchainState {
    pub ext: ash::extensions::khr::Swapchain,
    pub swapchain: ash::vk::SwapchainKHR,
    pub images: Vec<Image>,
    pub image_views: Vec<ImageView>,
    pub framebuffers: Vec<Framebuffer>,
    pub work_fences: Vec<Fence>,
    pub sem_work_done: Vec<Semaphore>,
    pub sem_img_available: Vec<Semaphore>,
    pub cmd_buffers: Vec<CommandBuffer>,
    pub image_index: u32,
    pub max_frames: u32,
}

impl VulkanSwapchainState {
    pub fn create_swapchain(
        ext: &ash::extensions::khr::Swapchain,
        surface: &VulkanSurfaceState,
        device: &Device,
        renderpass: RenderPass,
        queue: u32,
    ) -> (SwapchainKHR, Vec<Image>, Vec<ImageView>, Vec<Framebuffer>) {
        let swapchain = unsafe {
            ext.create_swapchain(
                &SwapchainCreateInfoKHR::builder()
                    .surface(surface.khr.surface)
                    .min_image_count(surface.image_count)
                    .image_format(surface.fmt.format)
                    .image_color_space(surface.fmt.color_space)
                    .image_extent(surface.image_size)
                    .image_array_layers(1)
                    .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
                    .image_sharing_mode(SharingMode::EXCLUSIVE)
                    .queue_family_indices(&[queue])
                    .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
                    .pre_transform(surface.transform)
                    .present_mode(surface.present_mode),
                None,
            )
            .expect("Failed to create swapchain!")
        };

        let images = unsafe { ext.get_swapchain_images(swapchain) }
            .expect("Failed to get swapchain images ...");

        let image_views = images
            .iter()
            .map(|img| unsafe {
                device
                    .create_image_view(
                        &ImageViewCreateInfo::builder()
                            .image(*img)
                            .view_type(ImageViewType::TYPE_2D)
                            .format(surface.fmt.format)
                            .components(
                                *ComponentMapping::builder()
                                    .r(ComponentSwizzle::IDENTITY)
                                    .g(ComponentSwizzle::IDENTITY)
                                    .b(ComponentSwizzle::IDENTITY)
                                    .a(ComponentSwizzle::IDENTITY),
                            )
                            .subresource_range(
                                *ImageSubresourceRange::builder()
                                    .aspect_mask(ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            ),
                        None,
                    )
                    .expect("Failed to create imageview for swapchain image")
            })
            .collect::<Vec<_>>();

        let framebuffers = image_views
            .iter()
            .map(|&img_view| {
                unsafe {
                    device.create_framebuffer(
                        &FramebufferCreateInfo::builder()
                            .render_pass(renderpass)
                            .attachments(&[img_view])
                            .width(surface.image_size.width)
                            .height(surface.image_size.height)
                            .layers(1),
                        None,
                    )
                }
                .expect("Failed to create framebuffer")
            })
            .collect::<Vec<_>>();

        (swapchain, images, image_views, framebuffers)
    }

    fn create_sync_objects(
        device: &Device,
        frames: u32,
    ) -> (Vec<Fence>, Vec<Semaphore>, Vec<Semaphore>) {
        let work_fences = (0..frames)
            .map(|_| unsafe {
                device
                    .create_fence(
                        &FenceCreateInfo::builder().flags(FenceCreateFlags::SIGNALED),
                        None,
                    )
                    .expect("Failed to create fence ...")
            })
            .collect::<Vec<_>>();

        let sem_work_done = (0..frames)
            .map(|_| unsafe {
                device
                    .create_semaphore(&SemaphoreCreateInfo::builder(), None)
                    .expect("Failed to create semaphore ...")
            })
            .collect::<Vec<_>>();

        let sem_image_available = (0..frames)
            .map(|_| unsafe {
                device
                    .create_semaphore(&SemaphoreCreateInfo::builder(), None)
                    .expect("Failed to create semaphore ...")
            })
            .collect::<Vec<_>>();

        (work_fences, sem_work_done, sem_image_available)
    }

    pub fn new(
        instance: &Instance,
        ds: &VulkanDeviceState,
        renderpass: RenderPass,
    ) -> Option<VulkanSwapchainState> {
        let ext = ash::extensions::khr::Swapchain::new(instance, &ds.device);

        let (swapchain, images, image_views, framebuffers) = Self::create_swapchain(
            &ext,
            &ds.surface,
            &ds.device,
            renderpass,
            ds.physical.queue_family_id,
        );

        let (work_fences, sem_work_done, sem_img_available) =
            Self::create_sync_objects(&ds.device, ds.surface.image_count);

        Some(VulkanSwapchainState {
            ext,
            swapchain,
            images,
            image_views,
            framebuffers,
            work_fences,
            sem_work_done,
            sem_img_available,
            cmd_buffers: unsafe {
                ds.device.allocate_command_buffers(
                    &CommandBufferAllocateInfo::builder()
                        .command_pool(ds.cmd_pool)
                        .command_buffer_count(ds.surface.image_count),
                )
            }
            .expect("Failed to allocate command buffers"),
            image_index: 0,
            max_frames: ds.surface.image_count,
        })
    }

    pub fn handle_suboptimal(&mut self, ds: &VulkanDeviceState, renderpass: RenderPass) {
        unsafe {
            ds.device
                .queue_wait_idle(ds.queue)
                .expect("Failed to wait queue idle");
            ds.device.device_wait_idle().expect("Failed to wait_idle()");

            self.framebuffers.iter().for_each(|fb| {
                ds.device.destroy_framebuffer(*fb, None);
            });

            self.image_views.iter().for_each(|iv| {
                ds.device.destroy_image_view(*iv, None);
            });

            self.ext.destroy_swapchain(self.swapchain, None);
        }

        let (swapchain, images, image_views, framebuffers) = Self::create_swapchain(
            &self.ext,
            &ds.surface,
            &ds.device,
            renderpass,
            ds.physical.queue_family_id,
        );

        self.swapchain = swapchain;
        self.images = images;
        self.image_views = image_views;
        self.framebuffers = framebuffers;
        self.image_index = 0;

        let (fences, work_done, img_avail) = Self::create_sync_objects(&ds.device, self.max_frames);

        self.work_fences = fences;
        self.sem_work_done = work_done;
        self.sem_img_available = img_avail;
    }
}

impl VulkanState {
    unsafe extern "system" fn debug_callback_stub(
        message_severity: ash::vk::DebugUtilsMessageSeverityFlagsEXT,
        _message_types: ash::vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const ash::vk::DebugUtilsMessengerCallbackDataEXT,
        _p_user_data: *mut std::os::raw::c_void,
    ) -> ash::vk::Bool32 {
        if message_severity.intersects(
            DebugUtilsMessageSeverityFlagsEXT::WARNING
                | DebugUtilsMessageSeverityFlagsEXT::ERROR
                | DebugUtilsMessageSeverityFlagsEXT::INFO,
        ) {
            log::debug!(
                "[Vulkan Debug::]\n\t{}",
                std::ffi::CStr::from_ptr((*p_callback_data).p_message)
                    .to_str()
                    .unwrap_or_default()
            );
        }
        ash::vk::FALSE
    }

    fn create_surface(
        entry: &Entry,
        instance: &Instance,
        wsi: WindowSystemIntegration,
    ) -> VulkanSurfaceKHRState {
        let xlib_surface = ash::extensions::khr::XlibSurface::new(entry, instance);

        let khr_surface = unsafe {
            xlib_surface.create_xlib_surface(
                &XlibSurfaceCreateInfoKHR::builder()
                    .dpy(wsi.native_disp as *mut ash::vk::Display)
                    .window(std::mem::transmute::<u64, ash::vk::Window>(wsi.native_win)),
                None,
            )
        }
        .expect("Failed to creale XLIB surface");

        VulkanSurfaceKHRState {
            ext: ash::extensions::khr::Surface::new(entry, instance),
            surface: khr_surface,
        }
    }

    fn pick_device(
        instance: &Instance,
        surface: VulkanSurfaceKHRState,
    ) -> Option<VulkanDeviceState> {
        let phys_devices = unsafe { instance.enumerate_physical_devices() }
            .expect("Failed to query physical devices");

        let mut phys_device: Option<VulkanPhysicalDeviceState> = None;
        let mut surface_state: Option<VulkanSurfaceState> = None;

        for pd in phys_devices {
            let pd_properties = unsafe { instance.get_physical_device_properties(pd) };

            if pd_properties.device_type != PhysicalDeviceType::DISCRETE_GPU
                && pd_properties.device_type != PhysicalDeviceType::INTEGRATED_GPU
            {
                log::info!(
                    "Rejecting device {} (not a GPU device)",
                    pd_properties.device_id
                );
                continue;
            }

            let pd_features = unsafe { instance.get_physical_device_features(pd) };

            if pd_features.multi_draw_indirect == 0 || !pd_features.geometry_shader == 0 {
                log::info!(
                    "Rejecting device {} (no geometry shader and/or MultiDrawIndirect)",
                    pd_properties.device_id
                );
                continue;
            }

            let queue_family_props =
                unsafe { instance.get_physical_device_queue_family_properties(pd) };

            let maybe_queue_id = queue_family_props
                .iter()
                .enumerate()
                .find(|(_, queue_props)| {
                    queue_props
                        .queue_flags
                        .contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
                })
                .map(|(queue_id, _)| queue_id);

            if maybe_queue_id.is_none() {
                log::info!(
                    "Rejecting device {} (no GRAPHICS + COMPUTE support)",
                    pd_properties.device_id
                );
                continue;
            }

            let queue_id = maybe_queue_id.unwrap() as u32;

            //
            // query surface support
            let device_surface_support = unsafe {
                surface
                    .ext
                    .get_physical_device_surface_support(pd, queue_id, surface.surface)
            }
            .expect("Failed to query device surface support");

            if !device_surface_support {
                log::info!(
                    "Rejecting device {} (does not support surface)",
                    pd_properties.device_id
                );
                continue;
            }

            let surface_caps = unsafe {
                surface
                    .ext
                    .get_physical_device_surface_capabilities(pd, surface.surface)
            }
            .expect("Failed to query device surface caps");

            if !surface_caps
                .supported_usage_flags
                .intersects(ImageUsageFlags::COLOR_ATTACHMENT)
            {
                log::info!(
                    "Rejecting device {} (surface does not support COLOR_ATTACHMENT)",
                    pd_properties.device_id
                );
                continue;
            }

            //
            // query surface formats
            let maybe_surface_format = unsafe {
                surface
                    .ext
                    .get_physical_device_surface_formats(pd, surface.surface)
                    .map(|surface_formats| {
                        surface_formats
                            .iter()
                            .find(|fmt| {
                                fmt.format == Format::B8G8R8A8_UNORM
                                    || fmt.format == Format::R8G8B8A8_UNORM
                            })
                            .copied()
                    })
                    .expect("Failed to query surface format")
            };

            if maybe_surface_format.is_none() {
                log::info!("Rejecting device {} (does not support surface format B8G8R8A8_UNORM/R8G8B8A8_UNORM)", pd_properties.device_id);
                continue;
            }

            //
            // query present mode
            let maybe_present_mode = unsafe {
                surface
                    .ext
                    .get_physical_device_surface_present_modes(pd, surface.surface)
                    .map(|present_modes| {
                        present_modes
                            .iter()
                            .find(|&&mode| {
                                mode == PresentModeKHR::MAILBOX || mode == PresentModeKHR::FIFO
                            })
                            .copied()
                    })
                    .expect("Failed to query present modes")
            };

            if maybe_present_mode.is_none() {
                log::info!(
                    "Rejecting device {} (does not support presentation mode MAILBOX/FIFO)",
                    pd_properties.device_id
                );
                continue;
            }

            let image_count = if surface_caps.max_image_count > 0 {
                (surface_caps.min_image_count + 1).max(surface_caps.max_image_count)
            } else {
                surface_caps.min_image_count + 1
            };

            let image_size = if surface_caps.current_extent.width == 0xFFFFFFFF {
                todo!("Handle this case");
            } else {
                surface_caps.current_extent
            };

            surface_state = Some(VulkanSurfaceState {
                khr: surface,
                image_count,
                image_size,
                transform: surface_caps.current_transform,
                fmt: maybe_surface_format.unwrap(),
                present_mode: maybe_present_mode.unwrap(),
            });

            let memory_properties = unsafe { instance.get_physical_device_memory_properties(pd) };

            phys_device = Some(VulkanPhysicalDeviceState {
                device: pd,
                properties: pd_properties,
                features: pd_features,
                queue_family_id: queue_id,
                memory_properties,
            });

            break;
        }

        if surface_state.is_none() || phys_device.is_none() {
            return None;
        }

        let surface_state = surface_state.unwrap();
        let phys_device = phys_device.unwrap();

        //
        // create logical device
        let enabled_device_extensions = [b"VK_KHR_swapchain\0".as_ptr() as *const c_char];
        let device = unsafe {
            instance.create_device(
                phys_device.device,
                &DeviceCreateInfo::builder()
                    .enabled_extension_names(&enabled_device_extensions)
                    .queue_create_infos(&[DeviceQueueCreateInfo::builder()
                        .queue_family_index(phys_device.queue_family_id)
                        .queue_priorities(&[1f32])
                        .build()]),
                None,
            )
        }
        .expect("Failed to create logical device ...");

        let queue = unsafe { device.get_device_queue(phys_device.queue_family_id, 0) };
        let cmd_pool = unsafe {
            device.create_command_pool(
                &CommandPoolCreateInfo::builder()
                    .queue_family_index(phys_device.queue_family_id)
                    .flags(
                        CommandPoolCreateFlags::TRANSIENT
                            | CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    ),
                None,
            )
        }
        .expect("Failed to create command pool");

        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &DescriptorPoolCreateInfo::builder()
                    .max_sets(4096)
                    .pool_sizes(&[*DescriptorPoolSize::builder()
                        .ty(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .descriptor_count(1024)]),
                None,
            )
        }
        .expect("Failed to create descriptor pool ...");

        Some(VulkanDeviceState {
            device,
            queue,
            physical: phys_device,
            surface: surface_state,
            cmd_pool,
            descriptor_pool,
        })
    }

    pub fn new(wsi: WindowSystemIntegration) -> Option<VulkanState> {
        let entry = Entry::linked();
        let instance_ver = entry
            .try_enumerate_instance_version()
            .expect("Failed to get Vulkan version")?;

        log::info!(
            "Vulkan version: {}.{}.{}",
            ash::vk::api_version_major(instance_ver),
            ash::vk::api_version_minor(instance_ver),
            ash::vk::api_version_patch(instance_ver)
        );

        let mut debug_utils_msg_create_info = DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | DebugUtilsMessageSeverityFlagsEXT::WARNING,
            )
            .message_type(
                DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(VulkanState::debug_callback_stub));

        let app_name_engine = [
            CString::new("Vulkan + Rust adventures").unwrap(),
            CString::new("Epic engine").unwrap(),
        ];

        let app_create_info = ApplicationInfo::builder()
            .api_version(ash::vk::make_api_version(0, 1, 3, 0))
            .application_name(app_name_engine[0].as_c_str())
            .application_version(1)
            .engine_name(app_name_engine[1].as_c_str())
            .engine_version(1);

        let enabled_layers = [b"VK_LAYER_KHRONOS_validation\0".as_ptr() as *const c_char];
        let enabled_instance_extensions = [
            b"VK_KHR_surface\0".as_ptr() as *const c_char,
            b"VK_EXT_debug_utils\0".as_ptr() as *const c_char,
            b"VK_KHR_xlib_surface\0".as_ptr() as *const c_char,
        ];

        unsafe {
            entry
                .enumerate_instance_extension_properties(None)
                .map(|instance_exts| {
                    instance_exts.iter().for_each(|ext| {
                        let ext_name = std::ffi::CStr::from_ptr(ext.extension_name.as_ptr())
                            .to_str()
                            .unwrap();
                        log::info!("[instance extension :: {}]", ext_name);
                    });
                })
                .expect("Failed to enumerate instance extensions ...");
        }

        let instance = unsafe {
            entry.create_instance(
                &InstanceCreateInfo::builder()
                    .application_info(&app_create_info)
                    .push_next(&mut debug_utils_msg_create_info)
                    .enabled_layer_names(&enabled_layers)
                    .enabled_extension_names(&enabled_instance_extensions),
                None,
            )
        }
        .expect("Failed to create instance");

        let (dbg, msgr) = unsafe {
            let dbg = DebugUtils::new(&entry, &instance);
            let msgr = dbg
                .create_debug_utils_messenger(&debug_utils_msg_create_info, None)
                .expect("Failed to create debug messenger");
            (dbg, msgr)
        };

        let surface_state = Self::create_surface(&entry, &instance, wsi);

        let device_state =
            Self::pick_device(&instance, surface_state).expect("Faile to pick device");

        let renderpass = Self::create_renderpass(&device_state);

        let swapchain_state =
            VulkanSwapchainState::new(&instance, &device_state, renderpass).unwrap();

        let resource_loader = ResourceLoadingState::new(&device_state);

        Some(VulkanState {
            dbg,
            msgr,
            entry,
            instance,
            ds: device_state,
            swapchain: swapchain_state,
            renderpass,
            resource_loader,
        })
    }

    fn create_renderpass(ds: &VulkanDeviceState) -> RenderPass {
        let attachment_descriptions = [AttachmentDescription::builder()
            .format(ds.surface.fmt.format)
            .samples(SampleCountFlags::TYPE_1)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::STORE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::PRESENT_SRC_KHR)
            .build()];

        let attachment_refs = [AttachmentReference::builder()
            .attachment(0)
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()];

        unsafe {
            ds.device.create_render_pass(
                &RenderPassCreateInfo::builder()
                    .attachments(&attachment_descriptions)
                    .subpasses(&[SubpassDescription::builder()
                        .color_attachments(&attachment_refs)
                        .pipeline_bind_point(PipelineBindPoint::GRAPHICS)
                        .build()]),
                None,
            )
        }
        .expect("Failed to create renderpass")
    }

    pub fn begin_rendering(&mut self, fb_size: Extent2D) -> FrameRenderContext {
        //
        // wait for previous submittted work
        unsafe {
            self.ds
                .device
                .wait_for_fences(
                    &[self.swapchain.work_fences[self.swapchain.image_index as usize]],
                    true,
                    u64::MAX,
                )
                .expect("Failed to wait for submits");
        }

        //
        // acquire next image

        let (swapchain_available_img_index, suboptimal) = unsafe {
            self.swapchain.ext.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                self.swapchain.sem_img_available[self.swapchain.image_index as usize],
                Fence::null(),
            )
        }
        .expect("Acquire image error ...");

        if suboptimal {
            todo!("handle swapchain suboptimal!");
        }

        unsafe {
            self.ds
                .device
                .reset_fences(&[self.swapchain.work_fences[self.swapchain.image_index as usize]])
                .expect("Failed to reset fence ...");
        }

        //
        // begin command buffer + renderpass

        unsafe {
            self.ds
                .device
                .reset_command_buffer(
                    self.swapchain.cmd_buffers[self.swapchain.image_index as usize],
                    CommandBufferResetFlags::empty(),
                )
                .expect("Failed to reset command buffer");

            self.ds
                .device
                .begin_command_buffer(
                    self.swapchain.cmd_buffers[self.swapchain.image_index as usize],
                    &CommandBufferBeginInfo::builder()
                        .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("Failed to begin command buffer ...");
        }

        FrameRenderContext {
            cmd_buff: self.swapchain.cmd_buffers[self.swapchain.image_index as usize],
            framebuffer: self.swapchain.framebuffers[swapchain_available_img_index as usize],
            fb_size,
            current_frame_id: self.swapchain.image_index,
        }
    }

    pub fn end_rendering(&mut self) {
        //
        // end command buffer + renderpass
        unsafe {
            self.ds
                .device
                .end_command_buffer(self.swapchain.cmd_buffers[self.swapchain.image_index as usize])
                .expect("Failed to end command buffer");
        }

        //
        // submit
        unsafe {
            self.ds
                .device
                .queue_submit(
                    self.ds.queue,
                    &[*SubmitInfo::builder()
                        .command_buffers(&[
                            self.swapchain.cmd_buffers[self.swapchain.image_index as usize]
                        ])
                        .wait_semaphores(&[
                            self.swapchain.sem_img_available[self.swapchain.image_index as usize]
                        ])
                        .wait_dst_stage_mask(&[PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                        .signal_semaphores(&[
                            self.swapchain.sem_work_done[self.swapchain.image_index as usize]
                        ])],
                    self.swapchain.work_fences[self.swapchain.image_index as usize],
                )
                .expect("Failed to submit work");

            match self.swapchain.ext.queue_present(
                self.ds.queue,
                &PresentInfoKHR::builder()
                    .image_indices(&[self.swapchain.image_index])
                    .swapchains(&[self.swapchain.swapchain])
                    .wait_semaphores(&[
                        self.swapchain.sem_work_done[self.swapchain.image_index as usize]
                    ]),
            ) {
                Err(e) => {
                    if e == ash::vk::Result::ERROR_OUT_OF_DATE_KHR {
                        log::info!("Swapchain out of date, recreating ...");
                        self.handle_surface_size_changed();
                        self.swapchain.handle_suboptimal(&self.ds, self.renderpass);
                    } else {
                        todo!("Handle this ...");
                    }
                }
                Ok(suboptimal) => {
                    if suboptimal {
                        log::info!("Swapchain suboptimal, recreating ...");
                        self.handle_surface_size_changed();
                        self.swapchain.handle_suboptimal(&self.ds, self.renderpass);
                    } else {
                        self.swapchain.image_index =
                            (self.swapchain.image_index + 1) % self.swapchain.max_frames;
                    }
                }
            };
        }
    }

    pub fn handle_surface_size_changed(&mut self) {
        let surface_caps = unsafe {
            self.ds
                .surface
                .khr
                .ext
                .get_physical_device_surface_capabilities(
                    self.ds.physical.device,
                    self.ds.surface.khr.surface,
                )
        }
        .expect("Failed to query surface caps");

        log::info!("Surface extent {:?}", surface_caps.current_extent);

        assert_ne!(surface_caps.current_extent.width, 0xFFFFFFFF);
        self.ds.surface.image_size = surface_caps.current_extent;
    }
}

pub struct FrameRenderContext {
    pub cmd_buff: CommandBuffer,
    pub framebuffer: Framebuffer,
    pub fb_size: Extent2D,
    pub current_frame_id: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, enum_iterator::Sequence)]
#[repr(u32)]
enum Coloring {
    BlackWhite,
    Smooth,
    Log,
    Hsv,
    Rainbow,
    Palette,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct FractalParameters {
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
}

impl std::default::Default for FractalParameters {
    fn default() -> Self {
        Self {
            screen_width: 1024,
            screen_height: 1024,
            iterations: 64,
            coloring: Coloring::BlackWhite,
            zoom: 1f32,
            ox: 0f32,
            oy: 0f32,
            fxmin: -Self::FRACTAL_HALF_WIDTH,
            fxmax: Self::FRACTAL_HALF_WIDTH,
            fymin: -Self::FRACTAL_HALF_HEIGHT,
            fymax: Self::FRACTAL_HALF_HEIGHT,
            escape_radius: 2,
        }
    }
}

impl FractalParameters {
    pub const MIN_TERATIONS: u32 = 4;
    pub const MAX_ITERATIONS: u32 = 2048;
    pub const ZOOM_IN_FACTOR: f32 = 0.85f32;
    pub const ZOOM_OUT_FACTOR: f32 = 2f32;
    const FRACTAL_XMIN: f32 = -2f32;
    const FRACTAL_XMAX: f32 = 2f32;
    const FRACTAL_YMIN: f32 = -1f32;
    const FRACTAL_YMAX: f32 = 1f32;
    const ESC_RADIUS_MIN: u32 = 2;
    const ESC_RADIUS_MAX: u32 = 4096;

    const FRACTAL_HALF_WIDTH: f32 =
        (FractalParameters::FRACTAL_XMAX - FractalParameters::FRACTAL_XMIN) * 0.5f32;
    const FRACTAL_HALF_HEIGHT: f32 =
        (FractalParameters::FRACTAL_YMAX - FractalParameters::FRACTAL_YMIN) * 0.5f32;

    fn center_moved(&mut self, cx: f32, cy: f32, zoom: bool) {
        let (cx, cy) = screen_coords_to_complex_coords(
            cx,
            cy,
            Self::FRACTAL_XMIN,
            Self::FRACTAL_XMAX,
            Self::FRACTAL_YMIN,
            Self::FRACTAL_YMAX,
            self.screen_width as f32,
            self.screen_height as f32,
        );

        //
        // also zoom when centering if CTRL is down
        if zoom {
            self.zoom *= Self::ZOOM_IN_FACTOR;
        }

        self.ox += cx * self.zoom;
        self.oy += cy * self.zoom;

        self.fxmin = self.ox - FractalParameters::FRACTAL_HALF_WIDTH * self.zoom;
        self.fxmax = self.ox + FractalParameters::FRACTAL_HALF_WIDTH * self.zoom;
        self.fymin = self.oy - FractalParameters::FRACTAL_HALF_HEIGHT * self.zoom;
        self.fymax = self.oy + FractalParameters::FRACTAL_HALF_HEIGHT * self.zoom;
    }

    fn zoom_out(&mut self) {
        self.zoom = (self.zoom * Self::ZOOM_OUT_FACTOR).min(1f32);
        self.fxmin = self.ox - FractalParameters::FRACTAL_HALF_WIDTH * self.zoom;
        self.fxmax = self.ox + FractalParameters::FRACTAL_HALF_WIDTH * self.zoom;
        self.fymin = self.oy - FractalParameters::FRACTAL_HALF_HEIGHT * self.zoom;
        self.fymax = self.oy + FractalParameters::FRACTAL_HALF_HEIGHT * self.zoom;
    }

    fn zoom_in(&mut self) {
        self.zoom = (self.zoom * Self::ZOOM_IN_FACTOR).max(0f32);
        self.fxmin = self.ox - FractalParameters::FRACTAL_HALF_WIDTH * self.zoom;
        self.fxmax = self.ox + FractalParameters::FRACTAL_HALF_WIDTH * self.zoom;
        self.fymin = self.oy - FractalParameters::FRACTAL_HALF_HEIGHT * self.zoom;
        self.fymax = self.oy + FractalParameters::FRACTAL_HALF_HEIGHT * self.zoom;
    }

    fn screen_resized(&mut self, width: u32, height: u32) {
        self.screen_width = width;
        self.screen_height = height;
    }

    fn increase_iterations(&mut self) {
        self.iterations = (self.iterations * 2).min(Self::MAX_ITERATIONS);
    }

    fn decrease_iterations(&mut self) {
        self.iterations = (self.iterations / 2).max(Self::MIN_TERATIONS);
    }

    fn next_color(&mut self) {
        self.coloring = next_cycle(&self.coloring).unwrap();
    }

    fn previous_color(&mut self) {
        self.coloring = previous_cycle(&self.coloring).unwrap();
    }
}

struct Fractal {
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    descriptor_set_layout: DescriptorSetLayout,
    buffer: UniqueBuffer,
    params: FractalParameters,
    ubo_params: UniqueBuffer,
    descriptor_ubo_buffer: DescriptorSet,
}

impl Fractal {
    fn new(vks: &VulkanState) -> Self {
        let (pipeline, pipeline_layout, descriptor_set_layout) =
            Self::create_graphics_pipeline(&vks.ds, vks.renderpass);

        let ubo_params = UniqueBuffer::new::<FractalParameters>(
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
            descriptor_set_layout,
            buffer: UniqueBuffer::new::<f32>(
                vks,
                BufferUsageFlags::VERTEX_BUFFER,
                MemoryPropertyFlags::DEVICE_LOCAL | MemoryPropertyFlags::HOST_VISIBLE,
                64,
            ),
            params: FractalParameters {
                screen_width: vks.ds.surface.image_size.width,
                screen_height: vks.ds.surface.image_size.height,
                ..Default::default()
            },
            ubo_params,
            descriptor_ubo_buffer,
        }
    }

    fn render(&mut self, vks: &VulkanState, context: &FrameRenderContext) {
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
        .write_data(&[self.params]);

        unsafe {
            // vks.ds.device.cmd_begin_render_pass(
            //     context.cmd_buff,
            //     &RenderPassBeginInfo::builder()
            //         .framebuffer(context.framebuffer)
            //         .render_area(render_area)
            //         .render_pass(vks.renderpass)
            //         .clear_values(&[ClearValue {
            //             color: ClearColorValue {
            //                 float32: [0f32, 0f32, 0f32, 1f32],
            //             },
            //         }]),
            //     SubpassContents::INLINE,
            // );

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

        let vsm = compile_shader_from_file("data/shaders/basic.vert", &ds.device).unwrap();
        let fsm = compile_shader_from_file("data/shaders/basic.frag", &ds.device).unwrap();

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

struct UniqueBuffer {
    device: *const Device,
    handle: Buffer,
    memory: DeviceMemory,
    item_aligned_size: usize,
}

impl std::ops::Drop for UniqueBuffer {
    fn drop(&mut self) {
        log::debug!(
            "Dropping buffer and memory {:?} -> {:?}",
            self.handle,
            self.memory
        );
        unsafe {
            (*self.device).destroy_buffer(self.handle, None);
            (*self.device).free_memory(self.memory, None);
        }
    }
}

impl UniqueBuffer {
    fn new<T: Sized>(
        ds: &VulkanState,
        usage: BufferUsageFlags,
        memory_flags: MemoryPropertyFlags,
        items: usize,
    ) -> Self {
        let align_size = if usage.intersects(BufferUsageFlags::UNIFORM_BUFFER) {
            ds.ds
                .physical
                .properties
                .limits
                .min_uniform_buffer_offset_alignment
        } else if usage.intersects(
            BufferUsageFlags::UNIFORM_TEXEL_BUFFER | BufferUsageFlags::STORAGE_TEXEL_BUFFER,
        ) {
            ds.ds
                .physical
                .properties
                .limits
                .min_texel_buffer_offset_alignment
        } else if usage.intersects(BufferUsageFlags::STORAGE_BUFFER) {
            ds.ds
                .physical
                .properties
                .limits
                .min_storage_buffer_offset_alignment
        } else {
            ds.ds.physical.properties.limits.non_coherent_atom_size
        } as usize;

        let item_aligned_size = round_up(size_of::<T>(), align_size);
        let size = item_aligned_size * items;

        let buffer = unsafe {
            ds.ds.device.create_buffer(
                &BufferCreateInfo::builder()
                    .size(size as DeviceSize)
                    .usage(usage)
                    .sharing_mode(SharingMode::EXCLUSIVE)
                    .queue_family_indices(&[ds.ds.physical.queue_family_id]),
                None,
            )
        }
        .expect("Failed to create buffer");

        let memory_req = unsafe { ds.ds.device.get_buffer_memory_requirements(buffer) };
        let mem_heap = choose_memory_heap(&memory_req, memory_flags, &ds.ds);

        let buffer_memory = unsafe {
            ds.ds.device.allocate_memory(
                &MemoryAllocateInfo::builder()
                    .allocation_size(memory_req.size)
                    .memory_type_index(mem_heap),
                None,
            )
        }
        .expect("Failed to allocate memory for buffer ...");

        unsafe {
            ds.ds
                .device
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .expect("Failed to bind memory for buffer");
        }

        log::debug!(
            "Create buffer and memory object {:?} -> {:?}",
            buffer,
            buffer_memory
        );

        Self {
            device: &ds.ds.device as *const _,
            handle: buffer,
            memory: buffer_memory,
            item_aligned_size,
        }
    }
}

fn round_up(num_to_round: usize, multiple: usize) -> usize {
    assert_ne!(multiple, 0);
    ((num_to_round + multiple - 1) / multiple) * multiple
}

struct UniqueBufferMapping<'a> {
    buffer: Buffer,
    buffer_mem: DeviceMemory,
    mapped_memory: *mut c_void,
    offset: usize,
    size: usize,
    alignment: usize,
    device: &'a Device,
}

impl<'a> UniqueBufferMapping<'a> {
    fn new(
        buf: &UniqueBuffer,
        ds: &'a VulkanDeviceState,
        offset: Option<usize>,
        size: Option<usize>,
    ) -> Self {
        let offset = offset.unwrap_or_default();
        if offset != 0 && size.is_none() {
            log::error!(
                "When mapping a buffer and specifying an offset a size must also be provided"
            );
            panic!("blyat!!");
        }

        let size = size.unwrap_or(WHOLE_SIZE as usize);

        let mapped_memory = unsafe {
            ds.device.map_memory(
                buf.memory,
                offset as u64,
                size as u64,
                MemoryMapFlags::empty(),
            )
        }
        .expect("Failed to map memory");

        Self {
            buffer: buf.handle,
            buffer_mem: buf.memory,
            mapped_memory,
            device: &ds.device,
            offset,
            alignment: buf.item_aligned_size,
            size,
        }
    }

    fn write_data<T: Sized + Copy>(&mut self, data: &[T]) {
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.mapped_memory as *mut T, data.len());
        }
    }

    fn write_data_with_offset<T: Sized + Copy>(&mut self, data: &[T], offset: isize) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                (self.mapped_memory as *mut T).offset(offset),
                data.len(),
            );
        }
    }
}

impl<'a> std::ops::Drop for UniqueBufferMapping<'a> {
    fn drop(&mut self) {
        unsafe {
            self.device
                .flush_mapped_memory_ranges(&[*MappedMemoryRange::builder()
                    .memory(self.buffer_mem)
                    .offset(self.offset as DeviceSize)
                    .size(self.size as DeviceSize)])
                .expect("Failed to flush mapped memory ...");

            self.device.unmap_memory(self.buffer_mem);
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

struct ImageWithMemory {
    device: *const Device,
    img: Image,
    memory: DeviceMemory,
}

impl std::ops::Drop for ImageWithMemory {
    fn drop(&mut self) {
        unsafe {
            (*self.device).free_memory(self.memory, None);
            (*self.device).destroy_image(self.img, None);
        }
    }
}
