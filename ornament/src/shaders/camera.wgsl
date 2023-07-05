struct Camera {
    origin: vec3<f32>,
    lower_left_corner: vec3<f32>,
    horizontal: vec3<f32>,
    vertical: vec3<f32>,
    u: vec3<f32>,
    v: vec3<f32>,
    w: vec3<f32>,
    lens_radius: f32
}

fn camera_get_ray(s: f32, t: f32) -> Ray {
    let rd = camera.lens_radius * random_in_unit_disk();
    let offset = camera.u * rd.x + camera.v * rd.y;
    return Ray(
        camera.origin + offset, 
        camera.lower_left_corner + s * camera.horizontal + t * camera.vertical - camera.origin - offset
    );
}