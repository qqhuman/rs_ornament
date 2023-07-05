use std::{mem, rc::Rc};

use wgpu::util::DeviceExt;

use crate::{bvh, gpu_structs, Scene};

use super::Settings;

pub const WORKGROUP_SIZE: u32 = 64;

pub struct Unit {
    pub device: Rc<wgpu::Device>,
    pub queue: Rc<wgpu::Queue>,
    pub target_buffer: TargetBuffer,
    workgroups: u32,
    rng_state_buffer: Buffer,
    storage_buffers_bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,

    // dynamic state
    dynamic_state: DynamicState,
    dynamic_state_buffer: UniformBuffer,
    dynamic_state_bind_group: wgpu::BindGroup,

    // camera/shapes/materials/constant state buffers
    constant_state_buffer: UniformBuffer,
    camera_buffer: UniformBuffer,
    materials_buffer: Buffer,
    normal_indices_buffer: Buffer,
    normals_buffer: Buffer,
    transforms_buffer: Buffer,
    bvh_structure_buffer: Buffer,
    buffers_bind_group: wgpu::BindGroup,
}

impl Unit {
    pub fn new(
        device: Rc<wgpu::Device>,
        queue: Rc<wgpu::Queue>,
        settings: &Settings,
        scene: &Scene,
    ) -> Self {
        let target_buffer = TargetBuffer::new(&device, settings.width, settings.height);
        let dynamic_state = DynamicState::new();
        let dynamic_state_buffer = UniformBuffer::new_from_bytes(
            &device,
            bytemuck::cast_slice(&[dynamic_state]),
            0,
            Some("compute_dynamic_state_buffer"),
        );

        let dynamic_state_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[dynamic_state_buffer.layout(wgpu::ShaderStages::COMPUTE)],
                label: Some("compute_dynamic_state_bind_group_layout"),
            });

        let dynamic_state_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &dynamic_state_bind_group_layout,
            entries: &[dynamic_state_buffer.binding()],
            label: Some("compute_dynamic_state_bind_group"),
        });

        let rng_seed = (0..(settings.width * settings.height)).collect::<Vec<_>>();
        let rng_state_buffer = Buffer::new_from_bytes(
            &device,
            bytemuck::cast_slice(rng_seed.as_slice()),
            2,
            Some("compute_rng_state_buffer"),
        );

        let storage_buffers_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("compute_storage_buffers_bind_group_layout"),
                entries: &[
                    target_buffer
                        .buffer
                        .layout(wgpu::ShaderStages::COMPUTE, false),
                    target_buffer
                        .accumulation_buffer
                        .layout(wgpu::ShaderStages::COMPUTE, false),
                    rng_state_buffer.layout(wgpu::ShaderStages::COMPUTE, false),
                ],
            });

        let storage_buffers_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_storage_buffers_bind_group"),
            layout: &storage_buffers_bind_group_layout,
            entries: &[
                target_buffer.buffer.binding(),
                target_buffer.accumulation_buffer.binding(),
                rng_state_buffer.binding(),
            ],
        });

        let bvh_tree = bvh::build(scene);
        let mut normals = bvh_tree.normals;
        let mut normal_indices = bvh_tree.normal_indices;
        let transforms = bvh_tree.transforms;
        let materials = bvh_tree.materials;
        let bvh_structure = bvh_tree.nodes;

        if normals.is_empty() {
            normals.push(Default::default())
        }

        if normal_indices.is_empty() {
            normal_indices.push(Default::default())
        }

        let materials_buffer = Buffer::new_from_bytes(
            &device,
            bytemuck::cast_slice(materials.as_slice()),
            0,
            Some("compute_materials_buffer"),
        );

        let constant_state_buffer = UniformBuffer::new_from_bytes(
            &device,
            bytemuck::cast_slice(&[gpu_structs::ConstantState::from(settings)]),
            1,
            Some("compute_constant_state_buffer"),
        );

        let camera_buffer = UniformBuffer::new_from_bytes(
            &device,
            bytemuck::cast_slice(&[gpu_structs::Camera::from(&scene.camera)]),
            2,
            Some("compute_camera_buffer"),
        );

        let normals_buffer = Buffer::new_from_bytes(
            &device,
            bytemuck::cast_slice(normals.as_slice()),
            3,
            Some("compute_normals_buffer"),
        );

        let normal_indices_buffer = Buffer::new_from_bytes(
            &device,
            bytemuck::cast_slice(normal_indices.as_slice()),
            4,
            Some("compute_normal_indices_buffer"),
        );

        let transforms_buffer = Buffer::new_from_bytes(
            &device,
            bytemuck::cast_slice(transforms.as_slice()),
            5,
            Some("compute_transforms_buffer"),
        );

        let bvh_structure_buffer = Buffer::new_from_bytes(
            &device,
            bytemuck::cast_slice(bvh_structure.as_slice()),
            6,
            Some("compute_bvh_structure_buffer"),
        );

        let buffers_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("compute_buffers_bind_group_layout"),
                entries: &[
                    materials_buffer.layout(wgpu::ShaderStages::COMPUTE, true),
                    constant_state_buffer.layout(wgpu::ShaderStages::COMPUTE),
                    camera_buffer.layout(wgpu::ShaderStages::COMPUTE),
                    normals_buffer.layout(wgpu::ShaderStages::COMPUTE, true),
                    normal_indices_buffer.layout(wgpu::ShaderStages::COMPUTE, true),
                    transforms_buffer.layout(wgpu::ShaderStages::COMPUTE, true),
                    bvh_structure_buffer.layout(wgpu::ShaderStages::COMPUTE, true),
                ],
            });

        let buffers_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_buffers_bind_group"),
            layout: &buffers_bind_group_layout,
            entries: &[
                materials_buffer.binding(),
                constant_state_buffer.binding(),
                camera_buffer.binding(),
                normals_buffer.binding(),
                normal_indices_buffer.binding(),
                transforms_buffer.binding(),
                bvh_structure_buffer.binding(),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute_pipeline_layout"),
            bind_group_layouts: &[
                &storage_buffers_bind_group_layout,
                &dynamic_state_bind_group_layout,
                &buffers_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let code = [
            include_str!("shaders/bvh.wgsl"),
            include_str!("shaders/pathtracer.wgsl"),
            include_str!("shaders/camera.wgsl"),
            include_str!("shaders/hitrecord.wgsl"),
            include_str!("shaders/material.wgsl"),
            include_str!("shaders/random.wgsl"),
            include_str!("shaders/ray.wgsl"),
            include_str!("shaders/sphere.wgsl"),
            include_str!("shaders/states.wgsl"),
            include_str!("shaders/transform.wgsl"),
            include_str!("shaders/utility.wgsl"),
        ]
        .join("\n");

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute_module"),
            source: wgpu::ShaderSource::Wgsl((&code).into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "main",
        });

        let pixels = settings.width * settings.height;
        let workgroups = (pixels / WORKGROUP_SIZE) + (pixels % WORKGROUP_SIZE);

        Self {
            device,
            queue,
            target_buffer,
            workgroups,
            rng_state_buffer,
            storage_buffers_bind_group,
            pipeline,

            dynamic_state,
            dynamic_state_buffer,
            dynamic_state_bind_group,

            constant_state_buffer,
            camera_buffer,
            materials_buffer,
            normals_buffer,
            normal_indices_buffer,
            transforms_buffer,
            bvh_structure_buffer,
            buffers_bind_group,
        }
    }

    pub fn reset(&mut self) {
        self.dynamic_state.reset();
    }

    pub fn update(&mut self, settings: &mut Settings, scene: &mut Scene) {
        let mut dirty = false;
        if scene.camera.dirty {
            dirty = true;
            scene.camera.dirty = false;
            self.queue.write_buffer(
                &self.camera_buffer.handle(),
                0,
                bytemuck::cast_slice(&[gpu_structs::Camera::from(&scene.camera)]),
            );
        }

        if settings.dirty {
            dirty = true;
            settings.dirty = false;
            self.queue.write_buffer(
                &self.constant_state_buffer.handle(),
                0,
                bytemuck::cast_slice(&[gpu_structs::ConstantState::from(&settings)]),
            );
        }

        if dirty {
            self.reset();
        }

        self.dynamic_state.next_iteration();
        self.queue.write_buffer(
            &self.dynamic_state_buffer.handle(),
            0,
            bytemuck::cast_slice(&[self.dynamic_state]),
        );
    }

    pub fn render_with_encoder(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Path Tracer Pass"),
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.storage_buffers_bind_group, &[]);
        pass.set_bind_group(1, &self.dynamic_state_bind_group, &[]);
        pass.set_bind_group(2, &self.buffers_bind_group, &[]);
        pass.dispatch_workgroups(self.workgroups, 1, 1);
        drop(pass);
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DynamicState {
    current_iteration: f32,
}

impl DynamicState {
    pub fn new() -> Self {
        Self {
            current_iteration: 0.0,
        }
    }

    pub fn next_iteration(&mut self) {
        self.current_iteration += 1.0;
    }

    pub fn reset(&mut self) {
        self.current_iteration = 0.0;
    }
}

pub struct Buffer {
    handle: wgpu::Buffer,
    binding_idx: u32,
}

impl Buffer {
    pub fn new_from_bytes(
        device: &wgpu::Device,
        bytes: &[u8],
        binding_idx: u32,
        label: Option<&str>,
    ) -> Self {
        let handle = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            label,
        });

        Self {
            handle,
            binding_idx,
        }
    }

    pub fn new(device: &wgpu::Device, size: u64, binding_idx: u32, label: Option<&str>) -> Self {
        let handle = device.create_buffer(&wgpu::BufferDescriptor {
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            size,
            label,
            mapped_at_creation: false,
        });

        Self {
            handle,
            binding_idx,
        }
    }

    pub fn layout(
        &self,
        visibility: wgpu::ShaderStages,
        read_only: bool,
    ) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: self.binding_idx,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    pub fn layout_with(
        &self,
        binding: u32,
        visibility: wgpu::ShaderStages,
        read_only: bool,
    ) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
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

    pub fn binding_with(&self, binding: u32) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: self.handle.as_entire_binding(),
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

    pub fn handle(&self) -> &wgpu::Buffer {
        &self.handle
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

pub struct TargetBuffer {
    buffer: Buffer,
    accumulation_buffer: Buffer,
}

impl TargetBuffer {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let size = width * height * (mem::size_of::<f32>() as u32) * 4;
        let size = size as u64;

        let buffer = Buffer::new(device, size, 0, Some("target buffer"));
        let accumulation_buffer = Buffer::new(device, size, 1, Some("accumulation buffer"));

        Self {
            buffer,
            accumulation_buffer,
        }
    }

    pub fn layout(
        &self,
        binding: u32,
        visibility: wgpu::ShaderStages,
        read_only: bool,
    ) -> wgpu::BindGroupLayoutEntry {
        self.buffer.layout_with(binding, visibility, read_only)
    }

    pub fn binding(&self, binding: u32) -> wgpu::BindGroupEntry<'_> {
        self.buffer.binding_with(binding)
    }
}
