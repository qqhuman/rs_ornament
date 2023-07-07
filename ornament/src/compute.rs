use std::{mem, rc::Rc};

use wgpu::util::DeviceExt;

use crate::{bvh, gpu_structs, Scene};

use super::Settings;

pub const WORKGROUP_SIZE: u32 = 64;

pub struct Unit {
    pub device: Rc<wgpu::Device>,
    pub queue: Rc<wgpu::Queue>,

    // framebuffer/accumulation buffer
    pub target_buffer: TargetBuffer,

    workgroups: u32,
    pipeline: wgpu::ComputePipeline,
    bind_groups: Vec<wgpu::BindGroup>,

    // rng buffer
    _rng_state_buffer: StorageBuffer,

    // dynamic state/constant state/camera
    dynamic_state: DynamicState,
    dynamic_state_buffer: UniformBuffer,
    constant_state_buffer: UniformBuffer,
    camera_buffer: UniformBuffer,

    // materials/bvh nodes
    _materials_buffer: StorageBuffer,
    _bvh_nodes_buffer: StorageBuffer,

    // normals/normals indices/transforms
    _normals_buffer: StorageBuffer,
    _normal_indices_buffer: StorageBuffer,
    _transforms_buffer: StorageBuffer,
}

impl Unit {
    pub fn new(
        device: Rc<wgpu::Device>,
        queue: Rc<wgpu::Queue>,
        settings: &Settings,
        scene: &Scene,
    ) -> Self {
        let bvh_tree = bvh::build(scene);
        let mut normals = bvh_tree.normals;
        let mut normal_indices = bvh_tree.normal_indices;
        let transforms = bvh_tree.transforms;
        let materials = bvh_tree.materials;
        let bvh_nodes = bvh_tree.nodes;

        if normals.is_empty() {
            normals.push(Default::default())
        }

        if normal_indices.is_empty() {
            normal_indices.push(Default::default())
        }

        let mut bind_group_layouts = vec![];
        let mut bind_groups = vec![];

        let (target_buffer, rng_state_buffer) = {
            let target_buffer = TargetBuffer::new(
                device.clone(),
                queue.clone(),
                settings.width,
                settings.height,
            );

            let rng_seed = (0..(settings.width * settings.height)).collect::<Vec<_>>();
            let rng_state_buffer = StorageBuffer::new_from_bytes(
                &device,
                bytemuck::cast_slice(rng_seed.as_slice()),
                2,
                Some("ornament_rng_state_buffer"),
            );

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("ornament_storage_buffers_bgl"),
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

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ornament_storage_buffers_bg"),
                layout: &bind_group_layout,
                entries: &[
                    target_buffer.buffer.binding(),
                    target_buffer.accumulation_buffer.binding(),
                    rng_state_buffer.binding(),
                ],
            });

            bind_group_layouts.push(bind_group_layout);
            bind_groups.push(bind_group);
            (target_buffer, rng_state_buffer)
        };

        let (dynamic_state, dynamic_state_buffer, constant_state_buffer, camera_buffer) = {
            let dynamic_state = DynamicState::new();
            let dynamic_state_buffer = UniformBuffer::new_from_bytes(
                &device,
                bytemuck::cast_slice(&[dynamic_state]),
                0,
                Some("ornament_dynamic_state_buffer"),
                false,
            );

            let constant_state_buffer = UniformBuffer::new_from_bytes(
                &device,
                bytemuck::cast_slice(&[gpu_structs::ConstantState::from(settings)]),
                1,
                Some("ornament_constant_state_buffer"),
                false,
            );

            let camera_buffer = UniformBuffer::new_from_bytes(
                &device,
                bytemuck::cast_slice(&[gpu_structs::Camera::from(&scene.camera)]),
                2,
                Some("ornament_camera_buffer"),
                false,
            );

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        dynamic_state_buffer.layout(wgpu::ShaderStages::COMPUTE),
                        constant_state_buffer.layout(wgpu::ShaderStages::COMPUTE),
                        camera_buffer.layout(wgpu::ShaderStages::COMPUTE),
                    ],
                    label: Some("ornament_dynstate__conststate__camera_bgl"),
                });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    dynamic_state_buffer.binding(),
                    constant_state_buffer.binding(),
                    camera_buffer.binding(),
                ],
                label: Some("ornament_dynstate__conststate__camera_bg"),
            });

            bind_group_layouts.push(bind_group_layout);
            bind_groups.push(bind_group);

            (
                dynamic_state,
                dynamic_state_buffer,
                constant_state_buffer,
                camera_buffer,
            )
        };

        let (materials_buffer, bvh_nodes_buffer) = {
            let materials_buffer = StorageBuffer::new_from_bytes(
                &device,
                bytemuck::cast_slice(materials.as_slice()),
                0,
                Some("ornament_materials_buffer"),
            );

            let bvh_nodes_buffer = StorageBuffer::new_from_bytes(
                &device,
                bytemuck::cast_slice(bvh_nodes.as_slice()),
                1,
                Some("ornament_bvh_nodes_buffer"),
            );

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("ornament_materials__bvh_nodes_bgl"),
                    entries: &[
                        materials_buffer.layout(wgpu::ShaderStages::COMPUTE, true),
                        bvh_nodes_buffer.layout(wgpu::ShaderStages::COMPUTE, true),
                    ],
                });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ornament_materials__bvh_nodes_bg"),
                layout: &bind_group_layout,
                entries: &[materials_buffer.binding(), bvh_nodes_buffer.binding()],
            });

            bind_group_layouts.push(bind_group_layout);
            bind_groups.push(bind_group);

            (materials_buffer, bvh_nodes_buffer)
        };

        let (normals_buffer, normal_indices_buffer, transforms_buffer) = {
            let normals_buffer = StorageBuffer::new_from_bytes(
                &device,
                bytemuck::cast_slice(normals.as_slice()),
                0,
                Some("ornament_normals_buffer"),
            );

            let normal_indices_buffer = StorageBuffer::new_from_bytes(
                &device,
                bytemuck::cast_slice(normal_indices.as_slice()),
                1,
                Some("ornament_normal_indices_buffer"),
            );

            let transforms_buffer = StorageBuffer::new_from_bytes(
                &device,
                bytemuck::cast_slice(transforms.as_slice()),
                2,
                Some("ornament_transforms_buffer"),
            );

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("ornament_normals__normals_indices__transforms_bgl"),
                    entries: &[
                        normals_buffer.layout(wgpu::ShaderStages::COMPUTE, true),
                        normal_indices_buffer.layout(wgpu::ShaderStages::COMPUTE, true),
                        transforms_buffer.layout(wgpu::ShaderStages::COMPUTE, true),
                    ],
                });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ornament_normals__normals_indices__transforms_bg"),
                layout: &bind_group_layout,
                entries: &[
                    normals_buffer.binding(),
                    normal_indices_buffer.binding(),
                    transforms_buffer.binding(),
                ],
            });

            bind_group_layouts.push(bind_group_layout);
            bind_groups.push(bind_group);

            (normals_buffer, normal_indices_buffer, transforms_buffer)
        };

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ornament_pipeline_layout"),
            bind_group_layouts: bind_group_layouts
                .iter()
                .collect::<Vec<&wgpu::BindGroupLayout>>()
                .as_slice(),
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
            label: Some("ornament_module"),
            source: wgpu::ShaderSource::Wgsl((&code).into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ornament_pipeline"),
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
            _rng_state_buffer: rng_state_buffer,
            pipeline,
            dynamic_state,
            dynamic_state_buffer,
            constant_state_buffer,
            camera_buffer,
            _materials_buffer: materials_buffer,
            _normals_buffer: normals_buffer,
            _normal_indices_buffer: normal_indices_buffer,
            _transforms_buffer: transforms_buffer,
            _bvh_nodes_buffer: bvh_nodes_buffer,
            bind_groups,
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
        for (pos, bg) in self.bind_groups.iter().enumerate() {
            pass.set_bind_group(pos as u32, bg, &[]);
        }
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

pub struct StorageBuffer {
    pub handle: wgpu::Buffer,
    binding_idx: u32,
}

impl StorageBuffer {
    pub fn new_from_bytes(
        device: &wgpu::Device,
        bytes: &[u8],
        binding_idx: u32,
        label: Option<&str>,
    ) -> Self {
        let handle = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
            label,
        });

        Self {
            handle,
            binding_idx,
        }
    }

    pub fn new(
        device: &wgpu::Device,
        size: u64,
        binding_idx: u32,
        label: Option<&str>,
        copy_src: bool,
    ) -> Self {
        let mut usage = wgpu::BufferUsages::STORAGE;
        if copy_src {
            usage = usage | wgpu::BufferUsages::COPY_SRC;
        }
        let handle = device.create_buffer(&wgpu::BufferDescriptor {
            usage,
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
        read_only: bool,
    ) -> Self {
        let mut usage = wgpu::BufferUsages::UNIFORM;
        if !read_only {
            usage = usage | wgpu::BufferUsages::COPY_DST;
        }

        let handle = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytes,
            usage,
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
    device: Rc<wgpu::Device>,
    queue: Rc<wgpu::Queue>,
    buffer: StorageBuffer,
    accumulation_buffer: StorageBuffer,
    map_buffer: wgpu::Buffer,
    size: u64,
}

impl TargetBuffer {
    pub fn new(device: Rc<wgpu::Device>, queue: Rc<wgpu::Queue>, width: u32, height: u32) -> Self {
        let size = width * height * (mem::size_of::<f32>() as u32) * 4;
        let size = size as u64;

        let buffer = StorageBuffer::new(&device, size, 0, Some("ornament_target_buffer"), true);
        let accumulation_buffer = StorageBuffer::new(
            &device,
            size,
            1,
            Some("ornament_target_accumulation_buffer"),
            false,
        );

        let map_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            size,
            label: Some("ornament_target_map_buffer"),
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            buffer,
            accumulation_buffer,
            map_buffer,
            size,
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

    pub async fn get_data(&self) -> wgpu::BufferView<'_> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("") });
        encoder.copy_buffer_to_buffer(&self.buffer.handle, 0, &self.map_buffer, 0, self.size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.map_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.receive().await.unwrap().unwrap();

        buffer_slice.get_mapped_range()
    }
}
