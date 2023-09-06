use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;

pub struct Timer {
    begin_frame: Instant,
    delta_time: Duration,
}

impl Timer {
    pub fn start() -> Self {
        Self {
            begin_frame: Instant::now(),
            delta_time: Duration::ZERO,
        }
    }

    pub fn end(mut self, frames: u32) {
        self.delta_time += self.begin_frame.elapsed();
        println!("Iterations: {:?} = {:?}", frames as u32, self.delta_time);
        self.delta_time = Duration::ZERO
    }
}

pub struct FpsCounter {
    begin_frame: Instant,
    frames: u32,
}

impl FpsCounter {
    pub fn new() -> Self {
        Self {
            begin_frame: Instant::now(),
            frames: 0,
        }
    }

    pub fn end_frames(&mut self, frames: u32) {
        self.frames += frames;
        let delta_time = self.begin_frame.elapsed();
        if delta_time >= Duration::new(1, 0) {
            let frames = self.frames as f64 / delta_time.as_secs_f64();
            println!("FPS: {:?}", frames as u32);
            self.frames = 0;
            self.begin_frame = Instant::now();
        }
    }
}

pub struct UniformBuffer {
    handle: wgpu::Buffer,
}

impl UniformBuffer {
    pub fn new_from_bytes(device: &wgpu::Device, bytes: &[u8], label: Option<&str>) -> Self {
        let handle = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            label,
        });

        Self { handle }
    }

    pub fn layout(
        &self,
        binding: u32,
        visibility: wgpu::ShaderStages,
    ) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    pub fn binding(&self, binding: u32) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: self.handle.as_entire_binding(),
        }
    }

    pub fn write(&self, queue: &wgpu::Queue, data: &[u8]) {
        queue.write_buffer(&self.handle, 0, data);
    }
}
