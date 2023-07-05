const pi: f32 = 3.1415926535897932385;

fn length_squared(e: vec3<f32>) -> f32 {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
}

const near_zero_s: f32 = 1e-8;
fn near_zero(e: vec3<f32>) -> bool {
    // Return true if the vector is close to zero in all dimensions.
    return (abs(e[0]) < near_zero_s) && (abs(e[1]) < near_zero_s) && (abs(e[2]) < near_zero_s);
}