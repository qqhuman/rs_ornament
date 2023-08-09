use std::{cell::OnceCell, mem, rc::Rc};

use wgpu::util::DeviceExt;

use crate::{bvh, gpu_structs, util::OnceCellExt, Error, Scene};

use super::State;

pub const WORKGROUP_SIZE: u32 = 256;

struct Pipelines {
    render_pipeline: wgpu::ComputePipeline,
    post_processing_pipeline: wgpu::ComputePipeline,
    render_and_post_processing_pipeline: wgpu::ComputePipeline,
    bind_groups: Vec<wgpu::BindGroup>,
}

pub(super) struct Unit {
    pub device: Rc<wgpu::Device>,
    pub queue: Rc<wgpu::Queue>,
    dynamic_state: DynamicState,
    buffers: Buffers,
    shader_module: wgpu::ShaderModule,
    pipelines: OnceCell<Pipelines>,
    width: u32,
    height: u32,
}

impl Unit {
    pub fn create(
        device: Rc<wgpu::Device>,
        queue: Rc<wgpu::Queue>,
        scene: &Scene,
        state: &State,
    ) -> Result<Self, Error> {
        let bvh_tree = bvh::build(scene)?;
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

        let dynamic_state = DynamicState::new();
        let (dynamic_state_buffer, constant_state_buffer, camera_buffer) = {
            let dynamic_state_buffer = UniformBuffer::new_from_bytes(
                &device,
                bytemuck::cast_slice(&[dynamic_state]),
                Some("ornament_dynamic_state_buffer"),
                false,
            );

            let constant_state_buffer = UniformBuffer::new_from_bytes(
                &device,
                bytemuck::cast_slice(&[gpu_structs::ConstantState::from(state)]),
                Some("ornament_constant_state_buffer"),
                false,
            );

            let camera_buffer = UniformBuffer::new_from_bytes(
                &device,
                bytemuck::cast_slice(&[gpu_structs::Camera::from(&scene.camera)]),
                Some("ornament_camera_buffer"),
                false,
            );
            (dynamic_state_buffer, constant_state_buffer, camera_buffer)
        };
        let buffers = Buffers {
            target: OnceCell::new(),
            dynamic_state: dynamic_state_buffer,
            constant_state: constant_state_buffer,
            camera: camera_buffer,
            materials: Self::create_materials_buffer(device.clone(), materials),
            bvh: BvhBuffers {
                normals: Self::create_normals_buffer(device.clone(), normals),
                normal_indices: Self::create_normal_indices_buffer(device.clone(), normal_indices),
                transforms: Self::create_transforms_buffer(device.clone(), transforms),
                nodes: Self::create_bvh_nodes_buffer(device.clone(), bvh_nodes),
            },
        };

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

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ornament_module"),
            source: wgpu::ShaderSource::Wgsl((&code).into()),
        });

        Ok(Self {
            device,
            queue,
            dynamic_state,
            buffers,
            shader_module,
            pipelines: OnceCell::new(),
            width: state.width,
            height: state.height,
        })
    }

    fn create_materials_buffer(
        device: Rc<wgpu::Device>,
        materials: Vec<gpu_structs::Material>,
    ) -> StorageBuffer {
        StorageBuffer::new_from_bytes(
            &device,
            bytemuck::cast_slice(materials.as_slice()),
            Some("ornament_materials_buffer"),
        )
    }

    fn create_bvh_nodes_buffer(
        device: Rc<wgpu::Device>,
        bvh_nodes: Vec<bvh::Node>,
    ) -> StorageBuffer {
        StorageBuffer::new_from_bytes(
            &device,
            bytemuck::cast_slice(bvh_nodes.as_slice()),
            Some("ornament_bvh_nodes_buffer"),
        )
    }

    fn create_normals_buffer(
        device: Rc<wgpu::Device>,
        normals: Vec<gpu_structs::Vector4>,
    ) -> StorageBuffer {
        StorageBuffer::new_from_bytes(
            &device,
            bytemuck::cast_slice(normals.as_slice()),
            Some("ornament_normals_buffer"),
        )
    }

    fn create_normal_indices_buffer(
        device: Rc<wgpu::Device>,
        normal_indices: Vec<u32>,
    ) -> StorageBuffer {
        StorageBuffer::new_from_bytes(
            &device,
            bytemuck::cast_slice(normal_indices.as_slice()),
            Some("ornament_normal_indices_buffer"),
        )
    }

    fn create_transforms_buffer(
        device: Rc<wgpu::Device>,
        transforms: Vec<gpu_structs::Transform>,
    ) -> StorageBuffer {
        StorageBuffer::new_from_bytes(
            &device,
            bytemuck::cast_slice(transforms.as_slice()),
            Some("ornament_transforms_buffer"),
        )
    }

    fn get_or_create_target_buffer(&self) -> &TargetBuffer {
        self.buffers.target.get_or_init(|| {
            TargetBuffer::new(
                self.device.clone(),
                self.queue.clone(),
                self.width,
                self.height,
            )
        })
    }

    fn get_workgoups(&self) -> u32 {
        self.get_or_create_target_buffer().workgroups
    }

    fn get_or_create_pipelines(&self) -> &Pipelines {
        self.pipelines.get_or_init(|| {
            let target_buffer = self.get_or_create_target_buffer();
            let mut bind_group_layouts = vec![];
            let mut bind_groups = vec![];
            {
                let bind_group_layout =
                    self.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("ornament_target_bgl"),
                            entries: &[
                                target_buffer
                                    .buffer
                                    .layout(0, wgpu::ShaderStages::COMPUTE, false),
                                target_buffer.accumulation_buffer.layout(
                                    1,
                                    wgpu::ShaderStages::COMPUTE,
                                    false,
                                ),
                                target_buffer.rng_state_buffer.layout(
                                    2,
                                    wgpu::ShaderStages::COMPUTE,
                                    false,
                                ),
                            ],
                        });

                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ornament_target_bg"),
                    layout: &bind_group_layout,
                    entries: &[
                        target_buffer.buffer.binding(0),
                        target_buffer.accumulation_buffer.binding(1),
                        target_buffer.rng_state_buffer.binding(2),
                    ],
                });

                bind_group_layouts.push(bind_group_layout);
                bind_groups.push(bind_group);
            }
            {
                let bind_group_layout =
                    self.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            entries: &[
                                self.buffers
                                    .dynamic_state
                                    .layout(0, wgpu::ShaderStages::COMPUTE),
                                self.buffers
                                    .constant_state
                                    .layout(1, wgpu::ShaderStages::COMPUTE),
                                self.buffers.camera.layout(2, wgpu::ShaderStages::COMPUTE),
                            ],
                            label: Some("ornament_dynstate__conststate__camera_bgl"),
                        });

                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &bind_group_layout,
                    entries: &[
                        self.buffers.dynamic_state.binding(0),
                        self.buffers.constant_state.binding(1),
                        self.buffers.camera.binding(2),
                    ],
                    label: Some("ornament_dynstate__conststate__camera_bg"),
                });

                bind_group_layouts.push(bind_group_layout);
                bind_groups.push(bind_group);
            }

            {
                let bind_group_layout =
                    self.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("ornament_materials__bvh_nodes_bgl"),
                            entries: &[
                                self.buffers
                                    .materials
                                    .layout(0, wgpu::ShaderStages::COMPUTE, true),
                                self.buffers
                                    .bvh
                                    .nodes
                                    .layout(1, wgpu::ShaderStages::COMPUTE, true),
                            ],
                        });

                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ornament_materials__bvh_nodes_bg"),
                    layout: &bind_group_layout,
                    entries: &[
                        self.buffers.materials.binding(0),
                        self.buffers.bvh.nodes.binding(1),
                    ],
                });

                bind_group_layouts.push(bind_group_layout);
                bind_groups.push(bind_group);
            };

            {
                let bind_group_layout =
                    self.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("ornament_normals__normals_indices__transforms_bgl"),
                            entries: &[
                                self.buffers.bvh.normals.layout(
                                    0,
                                    wgpu::ShaderStages::COMPUTE,
                                    true,
                                ),
                                self.buffers.bvh.normal_indices.layout(
                                    1,
                                    wgpu::ShaderStages::COMPUTE,
                                    true,
                                ),
                                self.buffers.bvh.transforms.layout(
                                    2,
                                    wgpu::ShaderStages::COMPUTE,
                                    true,
                                ),
                            ],
                        });

                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ornament_normals__normals_indices__transforms_bg"),
                    layout: &bind_group_layout,
                    entries: &[
                        self.buffers.bvh.normals.binding(0),
                        self.buffers.bvh.normal_indices.binding(1),
                        self.buffers.bvh.transforms.binding(2),
                    ],
                });

                bind_group_layouts.push(bind_group_layout);
                bind_groups.push(bind_group);
            };

            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("ornament_pipeline_layout"),
                        bind_group_layouts: bind_group_layouts
                            .iter()
                            .collect::<Vec<&wgpu::BindGroupLayout>>()
                            .as_slice(),
                        push_constant_ranges: &[],
                    });

            let render_pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("ornament_render_pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &self.shader_module,
                        entry_point: "main_render",
                    });

            let post_processing_pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("ornament_post_processing_pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &self.shader_module,
                        entry_point: "main_post_processing",
                    });

            let render_and_post_processing_pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("ornament_render_and_post_processing_pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &self.shader_module,
                        entry_point: "main",
                    });

            Pipelines {
                render_pipeline,
                post_processing_pipeline,
                render_and_post_processing_pipeline,
                bind_groups,
            }
        })
    }

    pub fn target_buffer_layout(
        &self,
        binding: u32,
        visibility: wgpu::ShaderStages,
        read_only: bool,
    ) -> wgpu::BindGroupLayoutEntry {
        self.get_or_create_target_buffer()
            .layout(binding, visibility, read_only)
    }

    pub fn target_buffer_binding(&self, binding: u32) -> wgpu::BindGroupEntry<'_> {
        self.get_or_create_target_buffer().binding(binding)
    }

    pub fn get_target_buffer_len(&self) -> u32 {
        self.get_or_create_target_buffer().get_buffer_len()
    }

    pub async fn get_target_buffer(&self, dst: &mut [f32]) -> Result<(), Error> {
        self.get_or_create_target_buffer().get_buffer(dst).await
    }

    pub fn set_target_buffer_resolution(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.pipelines.invalidate();
        self.buffers.target.invalidate();
    }

    pub fn reset(&mut self) {
        self.dynamic_state.reset();
    }

    pub fn update(&mut self, state: &mut State, scene: &mut Scene) {
        let mut dirty = false;
        if scene.camera.dirty {
            dirty = true;
            scene.camera.dirty = false;
            self.buffers.camera.write(
                &self.queue,
                bytemuck::cast_slice(&[gpu_structs::Camera::from(&scene.camera)]),
            );
        }

        if state.dirty {
            dirty = true;
            state.dirty = false;
            self.buffers.constant_state.write(
                &self.queue,
                bytemuck::cast_slice(&[gpu_structs::ConstantState::from(&state)]),
            );
        }

        if dirty {
            self.reset();
        }

        self.dynamic_state.next_iteration();
        self.buffers
            .dynamic_state
            .write(&self.queue, bytemuck::cast_slice(&[self.dynamic_state]));
    }

    pub fn render(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ornament_render_encoder"),
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ornament_render_pass"),
        });

        let pipelines = self.get_or_create_pipelines();
        pass.set_pipeline(&pipelines.render_pipeline);
        for (pos, bg) in pipelines.bind_groups.iter().enumerate() {
            pass.set_bind_group(pos as u32, bg, &[]);
        }
        pass.dispatch_workgroups(self.get_workgoups(), 1, 1);
        drop(pass);
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    pub fn post_processing(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ornament_post_processing_encoder"),
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ornament_post_processing_pass"),
        });

        let pipelines = self.get_or_create_pipelines();
        pass.set_pipeline(&pipelines.post_processing_pipeline);
        for (pos, bg) in pipelines.bind_groups.iter().enumerate() {
            pass.set_bind_group(pos as u32, bg, &[]);
        }
        pass.dispatch_workgroups(self.get_workgoups(), 1, 1);
        drop(pass);
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    pub fn render_and_apply_post_processing(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ornament_render_and_post_processing_encoder"),
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ornament_render_and_post_processing_pass"),
        });

        let pipelines = self.get_or_create_pipelines();
        pass.set_pipeline(&pipelines.render_and_post_processing_pipeline);
        for (pos, bg) in pipelines.bind_groups.iter().enumerate() {
            pass.set_bind_group(pos as u32, bg, &[]);
        }
        pass.dispatch_workgroups(self.get_workgoups(), 1, 1);
        drop(pass);
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
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
    handle: wgpu::Buffer,
}

impl StorageBuffer {
    pub fn new_from_bytes(device: &wgpu::Device, bytes: &[u8], label: Option<&str>) -> Self {
        let handle = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
            label,
        });

        Self { handle }
    }

    pub fn new(device: &wgpu::Device, size: u64, label: Option<&str>, copy_src: bool) -> Self {
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

        Self { handle }
    }

    pub fn layout(
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

    pub fn binding(&self, binding: u32) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: self.handle.as_entire_binding(),
        }
    }
}

pub struct UniformBuffer {
    handle: wgpu::Buffer,
}

impl UniformBuffer {
    pub fn new_from_bytes(
        device: &wgpu::Device,
        bytes: &[u8],
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

const TARGET_PIXEL_COMPONENTS: u32 = 4;
struct TargetBuffer {
    device: Rc<wgpu::Device>,
    queue: Rc<wgpu::Queue>,
    buffer: StorageBuffer,
    accumulation_buffer: StorageBuffer,
    rng_state_buffer: StorageBuffer,
    map_buffer: wgpu::Buffer,
    size: u64,
    width: u32,
    height: u32,
    workgroups: u32,
}

impl TargetBuffer {
    pub fn new(device: Rc<wgpu::Device>, queue: Rc<wgpu::Queue>, width: u32, height: u32) -> Self {
        let size = width * height * (mem::size_of::<f32>() as u32) * TARGET_PIXEL_COMPONENTS;
        let size = size as u64;

        let buffer = StorageBuffer::new(&device, size, Some("ornament_target_buffer"), true);
        let accumulation_buffer = StorageBuffer::new(
            &device,
            size,
            Some("ornament_target_accumulation_buffer"),
            false,
        );

        let rng_seed = (0..(width * height)).collect::<Vec<_>>();
        let rng_state_buffer = StorageBuffer::new_from_bytes(
            &device,
            bytemuck::cast_slice(rng_seed.as_slice()),
            Some("ornament_rng_state_buffer"),
        );

        let map_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            size,
            label: Some("ornament_target_map_buffer"),
            mapped_at_creation: false,
        });

        let pixels = width * height;
        let workgroups = (pixels / WORKGROUP_SIZE) + (pixels % WORKGROUP_SIZE);

        Self {
            device,
            queue,
            buffer,
            accumulation_buffer,
            rng_state_buffer,
            map_buffer,
            size,
            width,
            height,
            workgroups,
        }
    }

    pub fn layout(
        &self,
        binding: u32,
        visibility: wgpu::ShaderStages,
        read_only: bool,
    ) -> wgpu::BindGroupLayoutEntry {
        self.buffer.layout(binding, visibility, read_only)
    }

    pub fn binding(&self, binding: u32) -> wgpu::BindGroupEntry<'_> {
        self.buffer.binding(binding)
    }

    pub fn get_buffer_len(&self) -> u32 {
        self.width * self.height * TARGET_PIXEL_COMPONENTS
    }

    pub async fn get_buffer(&self, dst: &mut [f32]) -> Result<(), Error> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ornament_encoder_get_target_array"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer.handle, 0, &self.map_buffer, 0, self.size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.map_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.receive()
            .await
            .unwrap()
            .map_err(|_| Error::ReciveBuffer)?;

        let data = buffer_slice.get_mapped_range();
        unsafe {
            dst.copy_from_slice(std::slice::from_raw_parts(
                data.as_ptr() as *const f32,
                data.len() / TARGET_PIXEL_COMPONENTS as usize,
            ))
        }
        drop(data);
        self.map_buffer.unmap();
        Ok(())
    }
}

struct Buffers {
    target: OnceCell<TargetBuffer>,
    dynamic_state: UniformBuffer,
    constant_state: UniformBuffer,
    camera: UniformBuffer,
    materials: StorageBuffer,
    bvh: BvhBuffers,
}

struct BvhBuffers {
    nodes: StorageBuffer,
    normals: StorageBuffer,
    normal_indices: StorageBuffer,
    transforms: StorageBuffer,
}
