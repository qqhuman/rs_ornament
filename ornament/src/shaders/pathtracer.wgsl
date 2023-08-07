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

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id : vec3<u32>) {
    if invocation_id.x < arrayLength(&accumulation_buffer) {
        // init data
        init_rng_state(invocation_id.x);
        let dimensions = vec2<u32>(constant_state.width, constant_state.height);
        let xy = vec2<u32>(invocation_id.x % dimensions.x, invocation_id.x / dimensions.x);
        
        // trace ray
        let u = (f32(xy.x) + random_f32()) / f32(dimensions.x - 1u);
        let v = (f32(xy.y) + random_f32()) / f32(dimensions.y - 1u);
        let r = camera_get_ray(u, v);
        var rgb = ray_color(r);
        rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        var accumulated_rgba = vec4<f32>(rgb, 1.0);
        
        if dynamic_state.current_iteration > 1.0 {
            accumulated_rgba = accumulation_buffer[invocation_id.x] + accumulated_rgba;
        }

        var rgba = accumulated_rgba / dynamic_state.current_iteration;
        rgba.x = pow(rgba.x, constant_state.inverted_gamma);
        rgba.y = pow(rgba.y, constant_state.inverted_gamma);
        rgba.z = pow(rgba.z, constant_state.inverted_gamma);

        // store data
        accumulation_buffer[invocation_id.x] = accumulated_rgba;
        if constant_state.flip_y < 1u {
            framebuffer[invocation_id.x] = rgba;
        } else {
            let y_flipped = dimensions.y - xy.y - 1u;
            framebuffer[dimensions.x * y_flipped + xy.x] = rgba;
        }
        save_rng_state(invocation_id.x);
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