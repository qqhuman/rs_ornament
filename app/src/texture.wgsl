@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let u = (vertex_index << 1u) & 2u;
    let v = vertex_index & 2u;
    let uv = vec2<f32>(f32(u), f32(v));
    return vec4<f32>(uv * 2.0 + -1.0, 0.0, 1.0);
}

@group(0) @binding(0) var<storage, read> framebuffer : array<vec4<f32>>;
@group(0) @binding(1) var<uniform> dimensions: vec2<u32>;

// Fragment shader
@fragment
fn fs_main(@builtin(position) coord_in: vec4<f32>) -> @location(0) vec4<f32> {
    let xy = vec2<u32>(floor(coord_in.xy));
    if all(xy < dimensions) {
        let buffer_index = dimensions.x * xy.y + xy.x;
        return framebuffer[buffer_index];
    }
    else {
        return vec4<f32>(0.0, 0.0, 1.0, 1.0);
    }
}