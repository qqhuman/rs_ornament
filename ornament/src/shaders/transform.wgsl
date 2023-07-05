fn transform_point(transform_id: u32, point: vec3<f32>) -> vec3<f32> {
    let t = transforms[transform_id];
    let p = t * vec4<f32>(point, 1.0);
    return p.xyz;
}

fn transform_ray(transform_id: u32, ray: Ray) -> Ray {
    let inversed_t = transforms[transform_id];

    let o = inversed_t * vec4<f32>(ray.origin, 1.0);
    let d = inversed_t * vec4<f32>(ray.direction, 0.0);

    return Ray(o.xyz, d.xyz);
}

fn transform_normal(transform_id: u32, normal :vec3<f32>) -> vec3<f32> {
    let inversed_t = transforms[transform_id];
    let n = transpose(inversed_t) * vec4<f32>(normal, 0.0);
    return n.xyz;
}