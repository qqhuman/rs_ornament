const WIDTH: u32 = 1920 / 2;
const HEIGHT: u32 = 1080 / 2;
const DEPTH: u32 = 10;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let backens = wgpu::Backends::DX12;
    let limits = match backens {
        _ if cfg!(target_arch = "wasm32") => wgpu::Limits::downlevel_webgl2_defaults(),
        wgpu::Backends::GL => wgpu::Limits::downlevel_defaults(),
        wgpu::Backends::VULKAN | wgpu::Backends::DX12 => {
            let mut limits = wgpu::Limits::default();
            // extend the max storage buffer size from 134MB to 1073MB
            limits.max_storage_buffer_binding_size = 128 << 24;
            limits.max_buffer_size = 1 << 32;
            limits
        }
        _ => panic!("check the limits for the backend"),
    };
    let mut fps_counter = util::FpsCounter::new();
    let scene = examples::random_scene_mix_meshes_and_spheres(WIDTH, HEIGHT);
    let mut path_tracer = ornament::Context::new(scene, WIDTH, HEIGHT, backens, limits)
        .await
        .unwrap();

    path_tracer.set_depth(DEPTH);
    path_tracer.set_flip_y(true);
    path_tracer.set_gamma(2.2);

    for _ in 0..100 {
        fps_counter.start_frame();
        path_tracer.render();
        fps_counter.end_frame();
    }

    let floats = path_tracer.get_target_array().await.unwrap();
    let bytes = floats
        .iter()
        .map(|f| f32::clamp(f * 255.0, 0.0, 255.0) as u8)
        .collect();
    let buffer =
        image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(WIDTH, HEIGHT, bytes).unwrap();
    buffer.save("image.png").unwrap();
}
