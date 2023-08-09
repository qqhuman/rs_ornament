@group(0) @binding(0) var<storage, read_write> framebuffer : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> accumulation_buffer: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> rng_state_buffer: array<u32>;

@group(1) @binding(0) var<uniform> dynamic_state: DynamicState;
@group(1) @binding(1) var<uniform> constant_state: ConstantState;
@group(1) @binding(2) var<uniform> camera: Camera;

@group(2) @binding(0) var<storage, read> materials: array<Material>;
@group(2) @binding(1) var<storage, read> bvh_nodes: array<BvhNode>;

@group(3) @binding(0) var<storage, read> normals: array<vec3<f32>>;
@group(3) @binding(1) var<storage, read> normal_indices: array<u32>;
@group(3) @binding(2) var<storage, read> transforms: array<mat4x4<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main_render(@builtin(global_invocation_id) inv_id: vec3<u32>) {
    let inv_id_x = inv_id.x;
    if inv_id_x >= arrayLength(&accumulation_buffer) {
        return;
    }

    init_rng_state(inv_id_x);
    let xy = vec2<u32>(inv_id_x % constant_state.width, inv_id_x / constant_state.width);
    accumulation_buffer[inv_id_x] = render(inv_id_x, xy);
    save_rng_state(inv_id_x);
}

@compute @workgroup_size(256, 1, 1)
fn main_post_processing(@builtin(global_invocation_id) inv_id: vec3<u32>) {
    let inv_id_x = inv_id.x;
    if inv_id_x >= arrayLength(&accumulation_buffer) {
        return;
    }

    init_rng_state(inv_id_x);
    let xy = vec2<u32>(inv_id_x % constant_state.width, inv_id_x / constant_state.width);
    post_processing(inv_id_x, xy, accumulation_buffer[inv_id_x]);
    save_rng_state(inv_id_x);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) inv_id: vec3<u32>) {
    let inv_id_x = inv_id.x;
    if inv_id_x >= arrayLength(&accumulation_buffer) {
        return;
    }

    init_rng_state(inv_id_x);
    let xy = vec2<u32>(inv_id_x % constant_state.width, inv_id_x / constant_state.width);
    let accumulated_rgba = render(inv_id_x, xy);
    accumulation_buffer[inv_id_x] = accumulated_rgba;
    post_processing(inv_id_x, xy, accumulated_rgba);
    save_rng_state(inv_id_x);
}

fn render(inv_id_x: u32, xy: vec2<u32>) -> vec4<f32> {
    let u = (f32(xy.x) + random_f32()) / f32(constant_state.width - 1u);
    let v = (f32(xy.y) + random_f32()) / f32(constant_state.height - 1u);
    // mock tracing
    // var rgb = vec3<f32>(u, v, 0.0);
    // var accumulated_rgba = vec4<f32>(rgb, 1.0);
    let r = camera_get_ray(u, v);
    var rgb = ray_color(r);
    rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    var accumulated_rgba = vec4<f32>(rgb, 1.0);
    
    if dynamic_state.current_iteration > 1.0 {
        accumulated_rgba = accumulation_buffer[inv_id_x] + accumulated_rgba;
    }

    return accumulated_rgba;
}

fn post_processing(inv_id_x: u32, xy: vec2<u32>, accumulated_rgba: vec4<f32>) {
    var rgba = accumulated_rgba / dynamic_state.current_iteration;
    rgba.x = pow(rgba.x, constant_state.inverted_gamma);
    rgba.y = pow(rgba.y, constant_state.inverted_gamma);
    rgba.z = pow(rgba.z, constant_state.inverted_gamma);
    if constant_state.flip_y < 1u {
        framebuffer[inv_id_x] = rgba;
    } else {
        let y_flipped = constant_state.height - xy.y - 1u;
        framebuffer[constant_state.width * y_flipped + xy.x] = rgba;
    }
}

fn ray_color(ray: Ray) -> vec3<f32> {
    var current_ray = ray;
    var throughout = vec3<f32>(1.0);
    var color = vec3<f32>(0.0);
    var scattered = Ray();

    for (var i = 0u; i < constant_state.depth; i = i + 1u) {
        var rec = HitRecord();
        if bvh_hit(current_ray, &rec) {
            var attenuation = vec3<f32>(0.0);
            rec.normal = normalize(rec.normal);
            if material_scatter(current_ray, rec, &attenuation, &scattered) {
                current_ray = scattered;
                throughout *= attenuation;
                if all(throughout < 0.0001) {
                    break;
                }
            } else {
                throughout = vec3<f32>(0.0);
                break;
            }
        } else {
            var unit_direction = normalize(current_ray.direction);
            var t = 0.5 * (unit_direction.y + 1.0);
            color = (1.0 - t) * vec3<f32>(1.0) + t * vec3<f32>(0.5, 0.7, 1.0);
            break;
        }
    }

    return throughout * color;
}