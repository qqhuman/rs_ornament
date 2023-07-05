struct Sphere {
    center: vec3<f32>,
    radius: f32,
    material_index: u32
}

const sphere_center: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const sphere_radius: f32 = 1.0;

fn sphere_hit(ray: Ray, t_min: f32, t_max: f32) -> f32 {
    let oc = ray.origin - sphere_center;
    let a = length_squared(ray.direction);
    let half_b = dot(oc, ray.direction);
    let c = length_squared(oc) - sphere_radius * sphere_radius;
    let discriminant = half_b * half_b - a * c;
    if discriminant < 0.0 { return t_max; }
    let sqrtd = sqrt(discriminant);

    var t = (-half_b - sqrtd) / a;
    if t < t_min || t_max < t {
        t = (-half_b + sqrtd) / a;
        if t < t_min || t_max < t {
            return t_max;
        }
    }

    return t;
}