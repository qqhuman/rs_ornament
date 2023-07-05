struct HitRecord {
    p: vec3<f32>,
    normal: vec3<f32>,
    material_index: u32,
    material_type: u32,
    t: f32,
    front_face: bool
}

fn hit_record_set_face_normal(hit: ptr<function, HitRecord>, ray: Ray, outward_normal: vec3<f32>) {
    (*hit).front_face = dot(ray.direction, outward_normal) < 0.0;
    if (*hit).front_face {
        (*hit).normal = outward_normal;
    } else {
        (*hit).normal = -outward_normal;
    }
}