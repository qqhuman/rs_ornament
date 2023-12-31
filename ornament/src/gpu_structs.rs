use std::cell::Ref;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Camera {
    pub origin: Point3,
    _padding1: [u32; 1],
    pub lower_left_corner: Point3,
    _padding2: [u32; 1],
    pub horizontal: Vector3,
    _padding3: [u32; 1],
    pub vertical: Vector3,
    _padding4: [u32; 1],
    pub u: Vector3,
    _padding5: [u32; 1],
    pub v: Vector3,
    _padding6: [u32; 1],
    pub w: Vector3,
    pub lens_radius: f32,
}

impl Camera {
    pub fn from(camera: &crate::Camera) -> Self {
        Self {
            origin: camera.origin.into(),
            lower_left_corner: camera.lower_left_corner.into(),
            horizontal: camera.horizontal.into(),
            vertical: camera.vertical.into(),
            u: camera.u.into(),
            v: camera.v.into(),
            w: camera.w.into(),
            lens_radius: camera.lens_radius,
            ..Default::default()
        }
    }
}

pub type Point3 = [f32; 3];
pub type Vector3 = [f32; 3];
pub type Vector4 = [f32; 4];
pub type Color = [f32; 3];

pub type Transform = [[f32; 4]; 4];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    pub albedo: Color,
    pub fuzz: f32,          // 12 byte offset
    pub ior: f32,           // 16 byte offset
    pub material_type: u32, // 20 byte offset
    _padding: [u32; 2],     // 24 byte offset
}

impl Material {
    pub fn from(material: Ref<crate::Material>) -> Self {
        Self {
            albedo: material.albedo.into(),
            fuzz: material.fuzz,
            ior: material.ior,
            material_type: material.material_type,
            ..Default::default()
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ConstantState {
    depth: u32,
    width: u32,
    height: u32,
    flip_y: u32,
    inverted_gamma: f32,
    ray_cast_epsilon: f32,
}

impl ConstantState {
    pub fn from(state: &crate::State) -> Self {
        Self {
            depth: state.depth,
            width: state.width,
            height: state.height,
            flip_y: if state.flip_y { 1 } else { 0 },
            inverted_gamma: state.inverted_gamma,
            ray_cast_epsilon: state.ray_cast_epsilon,
        }
    }
}
