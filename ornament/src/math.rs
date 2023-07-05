use cgmath::Transform;

pub fn degrees_to_radians(degrees: f32) -> f32 {
    return degrees * std::f32::consts::PI / 180.0;
}

pub fn transform_aabb(m: &cgmath::Matrix4<f32>, aabb: &Aabb) -> Aabb {
    let p0 = aabb.min;
    let p1 = cgmath::Point3::new(aabb.max.x, aabb.min.y, aabb.min.z);
    let p2 = cgmath::Point3::new(aabb.min.x, aabb.max.y, aabb.min.z);
    let p3 = cgmath::Point3::new(aabb.min.x, aabb.min.y, aabb.max.z);
    let p4 = cgmath::Point3::new(aabb.min.x, aabb.max.y, aabb.max.z);
    let p5 = cgmath::Point3::new(aabb.max.x, aabb.max.y, aabb.min.z);
    let p6 = cgmath::Point3::new(aabb.max.x, aabb.min.y, aabb.max.z);
    let p7 = aabb.max;

    let p0 = m.transform_point(p0);
    let p1 = m.transform_point(p1);
    let p2 = m.transform_point(p2);
    let p3 = m.transform_point(p3);
    let p4 = m.transform_point(p4);
    let p5 = m.transform_point(p5);
    let p6 = m.transform_point(p6);
    let p7 = m.transform_point(p7);

    let mut aabb = Aabb::new(p0, p0);
    aabb.grow(p1);
    aabb.grow(p2);
    aabb.grow(p3);
    aabb.grow(p4);
    aabb.grow(p5);
    aabb.grow(p6);
    aabb.grow(p7);
    aabb
}

#[derive(Copy, Clone)]
pub struct Aabb {
    pub min: cgmath::Point3<f32>,
    pub max: cgmath::Point3<f32>,
}

impl Aabb {
    pub fn new(min: cgmath::Point3<f32>, max: cgmath::Point3<f32>) -> Aabb {
        Aabb { min, max }
    }

    fn grow(&mut self, p: cgmath::Point3<f32>) {
        self.min = point3_min(self.min, p);
        self.max = point3_max(self.max, p);
    }
}

pub fn point3_min(a: cgmath::Point3<f32>, b: cgmath::Point3<f32>) -> cgmath::Point3<f32> {
    cgmath::Point3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
}

pub fn point3_max(a: cgmath::Point3<f32>, b: cgmath::Point3<f32>) -> cgmath::Point3<f32> {
    cgmath::Point3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
}
