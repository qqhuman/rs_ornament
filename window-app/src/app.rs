use std::rc::Rc;

use util::{FpsCounter, UniformBuffer};
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    keyboard::KeyCode,
    window::{Window, WindowBuilder},
};

use crate::controllers;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const RESIZABLE: bool = true;
const DEPTH: u32 = 10;
const ITERATIONS: u32 = 1;

pub async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window_size = PhysicalSize {
        width: WIDTH,
        height: HEIGHT,
    };
    let window = WindowBuilder::new()
        .with_inner_size(window_size)
        .with_resizable(RESIZABLE)
        .build(&event_loop)
        .unwrap();
    let scene = examples::random_scene_spheres(window_size.width as u32, window_size.height as u32);

    let mut state = State::new(window, scene).await;
    let mut fps_counter = FpsCounter::new();

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window.id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                state: ElementState::Pressed,
                                physical_key: KeyCode::Escape,
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
        }
        Event::RedrawRequested(window_id) if window_id == state.window().id() => {
            fps_counter.start_frame();
            state.update();
            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                // The system is out of memory, we should probably quit
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                Err(e) => eprintln!("{:?}", e),
            }
            fps_counter.end_frames(ITERATIONS);
        }
        Event::MainEventsCleared => {
            // RedrawRequested will only trigger once, unless we manually
            // request it.
            state.window().request_redraw();
        }
        _ => {}
    });
}

struct State {
    path_tracer: ornament::Context,
    camera_controller: controllers::Camera,
    surface: wgpu::Surface,
    device: Rc<wgpu::Device>,
    queue: Rc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,
    render_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    _dimensions_buffer: UniformBuffer,
}

impl State {
    async fn new(window: Window, scene: ornament::Scene) -> Self {
        let size = window.inner_size();
        let backend_priorities = vec![
            wgpu::Backends::DX12,
            wgpu::Backends::VULKAN,
            wgpu::Backends::DX11,
            wgpu::Backends::GL,
            wgpu::Backends::METAL,
        ];
        let wgpu_context = request_wgpu_context(backend_priorities, &window).await;

        let device = Rc::new(wgpu_context.device);
        let queue = Rc::new(wgpu_context.queue);
        let surface = wgpu_context.surface;
        let mut path_tracer = ornament::Context::from_device_and_queue(
            device.clone(),
            queue.clone(),
            scene,
            size.width,
            size.height,
        )
        .unwrap();

        let surface_caps = surface.get_capabilities(&wgpu_context.adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render_shader_module"),
            source: wgpu::ShaderSource::Wgsl(include_str!("texture.wgsl").into()),
        });

        let (width, height) = path_tracer.get_resolution();
        let dimensions_buffer = UniformBuffer::new_from_bytes(
            &device,
            bytemuck::cast_slice(&[width, height]),
            1,
            Some("render_dimensions_buffer"),
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render_bgl"),
            entries: &[
                path_tracer.target_layout(0, wgpu::ShaderStages::FRAGMENT, true),
                dimensions_buffer.layout(wgpu::ShaderStages::FRAGMENT),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bg"),
            layout: &bind_group_layout,
            entries: &[path_tracer.target_binding(0), dimensions_buffer.binding()],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        path_tracer.set_flip_y(true);
        path_tracer.set_depth(DEPTH);
        path_tracer.set_iterations(ITERATIONS);
        if !surface_format.is_srgb() {
            path_tracer.set_gamma(2.2);
        }

        Self {
            path_tracer,
            camera_controller: controllers::Camera::new(0.7),
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            bind_group,
            _dimensions_buffer: dimensions_buffer,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera_controller.process_events(event)
    }

    fn update(&mut self) {
        self.camera_controller
            .update_camera(&mut self.path_tracer.scene.camera);
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.path_tracer.render();

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
        drop(render_pass);

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

struct WgpuContext {
    surface: wgpu::Surface,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

async fn request_wgpu_context_with_backend(
    backend: wgpu::Backends,
    window: &Window,
    limits: wgpu::Limits,
) -> Option<WgpuContext> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: backend,
        dx12_shader_compiler: Default::default(),
    });

    let surface = unsafe { instance.create_surface(&window).ok()? };

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await?;

    let device_descriptor = wgpu::DeviceDescriptor {
        features: wgpu::Features::empty(),
        limits,
        label: None,
    };
    let (device, queue) = adapter
        .request_device(&device_descriptor, None)
        .await
        .ok()?;

    Some(WgpuContext {
        surface,
        adapter,
        device,
        queue,
    })
}

async fn request_wgpu_context(properties: Vec<wgpu::Backends>, window: &Window) -> WgpuContext {
    for b in properties {
        let limits = match b {
            wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL => {
                let mut limits = wgpu::Limits::default();
                // extend the max storage buffer size from 134MB to 1073MB
                limits.max_storage_buffer_binding_size = 128 << 24;
                limits.max_buffer_size = 1 << 32;
                limits
            }
            wgpu::Backends::DX11 | wgpu::Backends::GL => wgpu::Limits::downlevel_defaults(),
            _ => panic!("check the limits for the backend"),
        };

        match request_wgpu_context_with_backend(b, window, limits).await {
            Some(wgpu_context) => return wgpu_context,
            _ => {}
        }
    }

    panic!("couldn't find working backend")
}
