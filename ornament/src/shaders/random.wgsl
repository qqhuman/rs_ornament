var<private> rng_state: u32;

fn init_rng_state(invocation_id: u32) {
    rng_state = rng_state_buffer[invocation_id];
}

fn save_rng_state(invocation_id: u32) {
    rng_state_buffer[invocation_id] = rng_state;
}

fn random_u32() -> u32 {
    // PCG random number generator
    // Based on https://www.shadertoy.com/view/XlGcRh
    let old_state = rng_state + 747796405u + 2891336453u;
    let word = ((old_state >> ((old_state >> 28u) + 4u)) ^ old_state) * 277803737u;
    rng_state = (word >> 22u) ^ word;
    return rng_state;
}

fn random_f32() -> f32 {
    return f32(random_u32()) / f32(0xffffffffu);
}

fn random_f32_between(min: f32, max: f32) -> f32 {
    return min + (max - min) * random_f32();
}

fn random_vec3() -> vec3<f32> {
    return vec3<f32>(random_f32(), random_f32(), random_f32());
}

fn random_vec3_between(min: f32, max: f32) -> vec3<f32> {
    return vec3<f32>(random_f32_between(min, max), random_f32_between(min, max), random_f32_between(min, max));
}

// fn random_in_unit_sphere() -> vec3<f32> {
//     var p: vec3<f32>;
//     loop {
//         let temp_p = random_vec3_between(-1.0, 1.0);
//         if length_squared(temp_p) < 1.0 {
//             p = temp_p;
//             break;
//         }
//     }
//     return p;
// }

fn random_in_unit_sphere() -> vec3<f32> {
    let r = pow(random_f32(), 0.33333);
    let theta = pi * random_f32();
    let phi = 2.0 * pi * random_f32();

    let x = r * sin(theta) * cos(phi);
    let y = r * sin(theta) * sin(phi);
    let z = r * cos(theta);

    return vec3<f32>(x, y, z);
}

fn random_unit_vector() -> vec3<f32> {
    return normalize(random_in_unit_sphere());
}

fn random_in_hemisphere(normal: vec3<f32>) -> vec3<f32> {
    let in_unit_sphere = random_in_unit_sphere();
    if dot(in_unit_sphere, normal) > 0.0 { // In the same hemisphere as the normal
        return in_unit_sphere;
    }
    else {
        return -in_unit_sphere;
    }
}

// fn random_in_unit_disk() -> vec3<f32> {
//     var p: vec3<f32>;
//     loop {
//         let temp_p = vec3<f32>(random_f32_between(-1.0, 1.0), random_f32_between(-1.0, 1.0), 0.0);
//         if length_squared(temp_p) < 1.0 {
//             p = temp_p;
//             break;
//         }
//     }
//     return p;
// }

fn random_in_unit_disk() -> vec3<f32> {
    // Generate numbers uniformly in a disk:
    // https://stats.stackexchange.com/a/481559

    // r^2 is distributed as U(0, 1).
    let r = sqrt(random_f32());
    let alpha = 2f * pi * random_f32();

    let x = r * cos(alpha);
    let y = r * sin(alpha);

    return vec3<f32>(x, y, 0.0);
}