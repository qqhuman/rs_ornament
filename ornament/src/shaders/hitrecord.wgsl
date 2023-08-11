struct HitRecord {
    p: vec3<f32>,
    normal: vec3<f32>,
    material_index: u32,
    material_type: u32,
    t: f32,
    front_face: bool
}

fn hit_record_set_face_normal(hit: ptr<function, HitRecord>, ray: Ray, outward_normal: vec3<f32>) {
    if dot(ray.direction, outward_normal) > 0.0 {
        (*hit).normal = -outward_normal;
        (*hit).front_face = false;
    } else {
        (*hit).normal = outward_normal;
        (*hit).front_face = true;
    }
}