use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;

pub struct FpsCounter {
    begin_frame: Instant,
    delta_time: Duration,
    frames: usize,
}

impl FpsCounter {
    pub fn new() -> Self {
        Self {
            begin_frame: Instant::now(),
            delta_time: Duration::ZERO,
            frames: 0,
        }
    }

    pub fn start_frame(&mut self) {
        self.begin_frame = Instant::now();
        self.frames += 1;
    }

    pub fn end_frame(&mut self) {
        self.delta_time += self.begin_frame.elapsed();
        if self.delta_time >= Duration::new(1, 0) {
            let frames = self.frames as f64 / self.delta_time.as_secs_f64();
            println!("FPS: {:?}", frames as u32);
            self.frames = 0;
            self.delta_time = Duration::ZERO
        }
    }
}

pub struct UniformBuffer {
    handle: wgpu::Buffer,
    binding_idx: u32,
}

impl UniformBuffer {
    pub fn new_from_bytes(
        device: &wgpu::Device,
        bytes: &[u8],
        binding_idx: u32,
        label: Option<&str>,
    ) -> Self {
        let handle = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            label,
        });

        Self {
            handle,
            binding_idx,
        }
    }

    pub fn layout(&self, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: self.binding_idx,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    pub fn binding(&self) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding: self.binding_idx,
            resource: self.handle.as_entire_binding(),
        }
    }
}