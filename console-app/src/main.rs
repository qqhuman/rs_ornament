const WIDTH: u32 = 1920 / 2;
const HEIGHT: u32 = 1080 / 2;
const DEPTH: u32 = 10;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let mut fps_counter = util::FpsCounter::new();
    let scene = examples::random_scene_mix_meshes_and_spheres(WIDTH, HEIGHT);
    let mut context_properties = ornament::ContextProperties::new_with_priorities(vec![
        ornament::Backend::Dx12,
        ornament::Backend::Vulkan,
        ornament::Backend::Dx11,
        ornament::Backend::Metal,
        ornament::Backend::Gl,
    ]);
    // extend the max storage buffer size from 134MB to 1073MB
    context_properties.max_storage_buffer_binding_size = 128 << 24;
    context_properties.max_buffer_size = 1 << 32;
    let mut path_tracer = ornament::Context::create(scene, WIDTH, HEIGHT, context_properties)
        .await
        .unwrap();

    path_tracer.set_depth(DEPTH);
    path_tracer.set_flip_y(true);
    path_tracer.set_gamma(2.2);

    for _ in 0..1000 {
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
