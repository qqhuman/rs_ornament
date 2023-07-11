use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
    sync::Mutex,
};

use cgmath::{Array, EuclideanSpace, InnerSpace};
use math::{degrees_to_radians, point3_max, point3_min, Aabb};

mod bvh;
mod compute;
mod gpu_structs;
mod math;

pub type RcCell<T> = Rc<RefCell<T>>;
pub type Color = cgmath::Point3<f32>;

pub struct Context {
    compute_unit: compute::Unit,
    state: State,
    pub scene: Scene,
}

impl Context {
    pub async fn new(
        scene: Scene,
        width: u32,
        height: u32,
        backends: wgpu::Backends,
        limits: wgpu::Limits,
    ) -> Result<Context, Error> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            dx12_shader_compiler: Default::default(),
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(Error::RequestAdapter)?;

        let device_descriptor = wgpu::DeviceDescriptor {
            features: wgpu::Features::empty(),
            limits,
            label: None,
        };
        let (device, queue) = adapter
            .request_device(&device_descriptor, None)
            .await
            .map_err(|_| Error::RequestDevice)?;

        Self::from_device_and_queue(Rc::new(device), Rc::new(queue), scene, width, height)
    }

    pub fn from_device_and_queue(
        device: Rc<wgpu::Device>,
        queue: Rc<wgpu::Queue>,
        scene: Scene,
        width: u32,
        height: u32,
    ) -> Result<Self, Error> {
        let state = State::new(width, height);
        let compute_unit = compute::Unit::try_create(device, queue, &scene, &state)?;

        Ok(Self {
            compute_unit,
            scene,
            state,
        })
    }

    pub fn target_layout(
        &self,
        binding: u32,
        visibility: wgpu::ShaderStages,
        read_only: bool,
    ) -> wgpu::BindGroupLayoutEntry {
        self.compute_unit
            .target_buffer
            .layout(binding, visibility, read_only)
    }

    pub fn target_binding(&self, binding: u32) -> wgpu::BindGroupEntry<'_> {
        self.compute_unit.target_buffer.binding(binding)
    }

    pub async fn get_target_array(&self) -> Result<Vec<f32>, Error> {
        self.compute_unit.target_buffer.get_target_array().await
    }

    pub fn clear_buffer(&mut self) {
        self.compute_unit.reset();
    }

    pub fn set_flip_y(&mut self, flip_y: bool) {
        self.state.set_flip_y(flip_y);
    }

    pub fn get_flip_y(&self) -> bool {
        self.state.get_flip_y()
    }

    pub fn set_gamma(&mut self, gamma: f32) {
        self.state.set_gamma(gamma);
    }

    pub fn get_gamma(&self) -> f32 {
        self.state.get_gamma()
    }

    pub fn set_depth(&mut self, depth: u32) {
        self.state.set_depth(depth);
    }

    pub fn get_depth(&self) -> u32 {
        self.state.get_depth()
    }

    pub fn set_resolution(&mut self, width: u32, height: u32) {
        self.state.set_resolution(width, height);
    }

    pub fn get_resolution(&self) -> (u32, u32) {
        self.state.get_resolution()
    }

    pub fn render(&mut self) {
        let mut encoder =
            self.compute_unit
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Path Tracer Encoder"),
                });

        self.render_with_encoder(&mut encoder);
        self.compute_unit
            .queue
            .submit(std::iter::once(encoder.finish()));
        self.compute_unit.device.poll(wgpu::Maintain::Wait);
    }

    pub fn render_with_encoder(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.compute_unit.update(&mut self.state, &mut self.scene);
        self.compute_unit.render_with_encoder(encoder);
    }
}

struct State {
    width: u32,
    height: u32,
    depth: u32,
    flip_y: bool,
    inverted_gamma: f32,
    dirty: bool,
}

impl State {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            depth: 10,
            flip_y: false,
            inverted_gamma: 1.0,
            dirty: true,
        }
    }

    fn make_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn set_flip_y(&mut self, flip_y: bool) {
        self.flip_y = flip_y;
        self.make_dirty();
    }

    pub fn get_flip_y(&self) -> bool {
        self.flip_y
    }

    pub fn set_gamma(&mut self, gamma: f32) {
        self.inverted_gamma = 1.0 / gamma;
        self.make_dirty();
    }

    pub fn get_gamma(&self) -> f32 {
        1.0 / self.inverted_gamma
    }

    pub fn set_depth(&mut self, depth: u32) {
        self.depth = depth;
        self.make_dirty();
    }

    pub fn get_depth(&self) -> u32 {
        self.depth
    }

    pub fn set_resolution(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.make_dirty();
    }

    pub fn get_resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

pub struct Camera {
    origin: cgmath::Point3<f32>,
    lower_left_corner: cgmath::Point3<f32>,
    horizontal: cgmath::Vector3<f32>,
    vertical: cgmath::Vector3<f32>,
    u: cgmath::Vector3<f32>,
    v: cgmath::Vector3<f32>,
    w: cgmath::Vector3<f32>,
    lens_radius: f32,

    vfov: f32,
    focus_dist: f32,
    aspect_ratio: f32,
    lookfrom: cgmath::Point3<f32>,
    lookat: cgmath::Point3<f32>,
    vup: cgmath::Vector3<f32>,

    dirty: bool,
}

impl Camera {
    pub fn new(
        lookfrom: cgmath::Point3<f32>,
        lookat: cgmath::Point3<f32>,
        vup: cgmath::Vector3<f32>,
        aspect_ratio: f32,
        vfov: f32,
        aperture: f32,
        focus_dist: f32,
    ) -> Camera {
        let theta = degrees_to_radians(vfov);
        let h = f32::tan(theta / 2.0);
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (lookfrom - lookat).normalize();
        let u = cgmath::Vector3::cross(vup, w).normalize();
        let v = cgmath::Vector3::cross(w, u);

        let origin = lookfrom;
        let horizontal = focus_dist * viewport_width * u;
        let vertical = focus_dist * viewport_height * v;
        let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - focus_dist * w;

        let lens_radius = aperture / 2.0;
        Camera {
            origin,
            lower_left_corner,
            horizontal,
            vertical,
            u,
            v,
            w,
            lens_radius,

            focus_dist,
            vfov,
            aspect_ratio,
            lookfrom,
            lookat,
            vup,

            dirty: true,
        }
    }

    pub fn get_look_from(&self) -> cgmath::Point3<f32> {
        self.lookfrom
    }

    pub fn get_look_at(&self) -> cgmath::Point3<f32> {
        self.lookat
    }

    pub fn get_vup(&self) -> cgmath::Vector3<f32> {
        self.vup
    }

    pub fn set_look_at(
        &mut self,
        lookfrom: cgmath::Point3<f32>,
        lookat: cgmath::Point3<f32>,
        vup: cgmath::Vector3<f32>,
    ) {
        *self = Camera::new(
            lookfrom,
            lookat,
            vup,
            self.aspect_ratio,
            self.vfov,
            2.0 * self.lens_radius,
            self.focus_dist,
        );
    }
}

pub struct MeshInstance {
    id: ShapeId,
    mesh: RcCell<Mesh>,
    material: RcCell<Material>,
    transform: cgmath::Matrix4<f32>,
    aabb: Aabb,
}

impl MeshInstance {
    pub fn new(
        mesh: RcCell<Mesh>,
        material: RcCell<Material>,
        transform: cgmath::Matrix4<f32>,
    ) -> MeshInstance {
        let aabb = math::transform_aabb(&transform, &mesh.borrow().not_transformed_aabb);
        MeshInstance {
            id: get_next_id(),
            mesh,
            material,
            transform,
            aabb,
        }
    }
}

pub struct Sphere {
    id: ShapeId,
    material: RcCell<Material>,
    transform: cgmath::Matrix4<f32>,
    aabb: Aabb,
}

impl Sphere {
    pub fn new(center: cgmath::Point3<f32>, radius: f32, material: RcCell<Material>) -> Self {
        let transform = cgmath::Matrix4::from_translation(center.to_vec())
            * cgmath::Matrix4::from_scale(radius);
        let radius_vec = cgmath::Vector3::new(radius, radius, radius);
        Self {
            id: get_next_id(),
            material,
            transform,
            aabb: Aabb::new(center - radius_vec, center + radius_vec),
        }
    }
}

pub struct Mesh {
    id: ShapeId,
    vertices: Vec<cgmath::Point3<f32>>,
    vertex_indices: Vec<u32>,
    normals: Vec<cgmath::Vector3<f32>>,
    normal_indices: Vec<u32>,
    transform: cgmath::Matrix4<f32>,
    material: RcCell<Material>,
    bvh_id: u32,
    aabb: Aabb,
    not_transformed_aabb: Aabb,
}

impl Mesh {
    pub fn new(
        vertices: Vec<cgmath::Point3<f32>>,
        vertex_indices: Vec<u32>,
        normals: Vec<cgmath::Vector3<f32>>,
        normal_indices: Vec<u32>,
        transform: cgmath::Matrix4<f32>,
        material: RcCell<Material>,
    ) -> Self {
        let mut min = cgmath::Point3 {
            x: f32::MAX,
            y: f32::MAX,
            z: f32::MAX,
        };
        let mut max = cgmath::Point3 {
            x: f32::MIN,
            y: f32::MIN,
            z: f32::MIN,
        };

        for mesh_triangle_index in 0..vertex_indices.len() / 3 {
            let v0 = vertices[vertex_indices[mesh_triangle_index * 3] as usize];
            let v1 = vertices[vertex_indices[mesh_triangle_index * 3 + 1] as usize];
            let v2 = vertices[vertex_indices[mesh_triangle_index * 3 + 2] as usize];
            let aabb = Aabb::new(
                point3_min(point3_min(v0, v1), v2),
                point3_max(point3_max(v0, v1), v2),
            );

            min = point3_min(min, aabb.min);
            max = point3_max(max, aabb.max);
        }

        let not_transformed_aabb = Aabb::new(min, max);
        let aabb = math::transform_aabb(&transform, &not_transformed_aabb);

        Self {
            id: get_next_id(),
            vertices,
            vertex_indices,
            normals,
            normal_indices,
            transform,
            material,
            aabb,
            not_transformed_aabb,
            bvh_id: 0,
        }
    }

    pub fn sphere(center: cgmath::Point3<f32>, radius: f32, material: RcCell<Material>) -> Mesh {
        let mut vertices = vec![];
        let mut normals = vec![];
        let mut indices = vec![];
        let facing = 1.0f32;
        let h_segments: u32 = 60;
        let v_segments: u32 = 30;

        // Add the top vertex.
        vertices.push(cgmath::Point3::new(0.0, 1.0, 0.0));
        normals.push(cgmath::Vector3::new(0.0, facing, 0.0));

        for v in 0..v_segments {
            if v == 0 {
                continue;
            }
            let theta = v as f32 / v_segments as f32 * std::f32::consts::PI;
            let sin_theta = theta.sin();

            for h in 0..h_segments {
                let phi = h as f32 / h_segments as f32 * std::f32::consts::PI * 2.0;
                let x = sin_theta * phi.sin() * 1.0;
                let z = sin_theta * phi.cos() * 1.0;
                let y = theta.cos() * 1.0;

                vertices.push(cgmath::Point3::new(x, y, z));
                normals.push((cgmath::Vector3::new(x, y, z) * facing).normalize());

                // Top triangle fan.
                if v == 1 {
                    indices.push(0);
                    indices.push(h + 1);
                    if h < h_segments - 1 {
                        indices.push(h + 2);
                    } else {
                        indices.push(1);
                    }
                }
                // Vertical slice.
                else {
                    let i = h + ((v - 1) * h_segments) + 1;
                    let j = i - h_segments;
                    let k = if h < h_segments - 1 {
                        j + 1
                    } else {
                        j - (h_segments - 1)
                    };
                    let l = if h < h_segments - 1 {
                        i + 1
                    } else {
                        i - (h_segments - 1)
                    };

                    indices.push(j);
                    indices.push(i);
                    indices.push(k);
                    indices.push(k);
                    indices.push(i);
                    indices.push(l);
                }
            }
        }

        // Bottom vertex.
        vertices.push(cgmath::Point3::new(0.0, -1.0, 0.0));
        normals.push(cgmath::Vector3::new(0.0, -facing, 0.0));

        // Bottom triangle fan.
        let vertex_count = vertices.len() as u32;
        let end = vertex_count - 1;

        for h in 0..h_segments {
            let i = end - h_segments + h;

            indices.push(i);
            indices.push(end);

            if h < h_segments - 1 {
                indices.push(i + 1);
            } else {
                indices.push(end - h_segments);
            }
        }

        let transform = cgmath::Matrix4::from_translation(center.to_vec())
            * cgmath::Matrix4::from_scale(radius);
        Mesh::new(
            vertices,
            indices.clone(),
            normals,
            indices,
            transform,
            material,
        )
    }
}

enum MaterialType {
    Lambertian = 0,
    Metal = 1,
    Dielectric = 2,
}

pub struct Material {
    albedo: Color,
    fuzz: f32,
    ior: f32,
    material_type: u32,
    material_index: Option<u32>,
}

impl Material {
    pub fn lambertian(albedo: Color) -> RcCell<Material> {
        Rc::new(RefCell::new(Material {
            albedo,
            material_type: MaterialType::Lambertian as u32,

            fuzz: 0.0,
            ior: 0.0,
            material_index: None,
        }))
    }

    pub fn metal(albedo: Color, fuzz: f32) -> RcCell<Material> {
        Rc::new(RefCell::new(Material {
            albedo,
            fuzz,
            material_type: MaterialType::Metal as u32,

            ior: 0.0,
            material_index: None,
        }))
    }

    pub fn dielectric(ior: f32) -> RcCell<Material> {
        Rc::new(RefCell::new(Material {
            ior,
            material_type: MaterialType::Dielectric as u32,

            albedo: cgmath::Point3::from_value(0.0),
            fuzz: 0.0,
            material_index: None,
        }))
    }
}

pub struct Scene {
    spheres: HashMap<ShapeId, RcCell<Sphere>>,
    meshes: HashMap<ShapeId, RcCell<Mesh>>,
    mesh_instances: HashMap<ShapeId, RcCell<MeshInstance>>,
    only_referenced_meshes: HashSet<ShapeId>,
    pub camera: Camera,
}

impl Scene {
    pub fn new(camera: Camera) -> Self {
        Self {
            spheres: HashMap::new(),
            meshes: HashMap::new(),
            mesh_instances: HashMap::new(),
            only_referenced_meshes: HashSet::new(),
            camera,
        }
    }

    pub fn add_sphere(&mut self, s: Sphere) -> RcCell<Sphere> {
        let id = s.id;
        let s = Rc::new(RefCell::new(s));
        self.spheres.insert(id, s.clone());
        s
    }

    pub fn add_mesh(&mut self, m: Mesh) -> RcCell<Mesh> {
        let id = m.id;
        if self.only_referenced_meshes.contains(&id) {
            self.only_referenced_meshes.remove(&id);
        }

        let m = Rc::new(RefCell::new(m));
        self.meshes.insert(id, m.clone());
        m
    }

    pub fn add_mesh_instances(&mut self, mi: MeshInstance) -> RcCell<MeshInstance> {
        if !self.meshes.contains_key(&mi.mesh.borrow().id) {
            let mesh = mi.mesh.borrow();
            self.meshes.insert(mesh.id, Rc::clone(&mi.mesh));
            self.only_referenced_meshes.insert(mesh.id);
        }

        let mi = Rc::new(RefCell::new(mi));
        self.mesh_instances.insert(mi.borrow().id, mi.clone());
        mi
    }
}

type ShapeId = usize;
static LAST_ID: Mutex<usize> = Mutex::new(0);
fn get_next_id() -> ShapeId {
    let mut last_id = LAST_ID.lock().unwrap();
    let id = *last_id;
    *last_id += 1;
    id
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Error {
    RequestAdapter,
    RequestDevice,
    SendBuffer,
    ReciveBuffer,
    BuildBvh,
    Unknown,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Error::Unknown => write!(f, "Unknown error occurred"),
            Error::RequestDevice => write!(f, "RequestDevice error occurred"),
            Error::SendBuffer => write!(f, "SendBuffer error occurred"),
            Error::ReciveBuffer => write!(f, "ReciveBuffer error occurred"),
            Error::RequestAdapter => write!(f, "RequestAdapter error occurred"),
            Error::BuildBvh => write!(f, "BuildBvh error occurred"),
        }
    }
}

impl std::error::Error for Error {}
