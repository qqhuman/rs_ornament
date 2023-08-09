const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const DEPTH: u32 = 10;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let scene = examples::random_scene_with_3_lucy(WIDTH as f32 / HEIGHT as f32);
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
    let mut path_tracer = ornament::Context::create(scene, context_properties)
        .await
        .unwrap();

    path_tracer.set_resolution(WIDTH, HEIGHT);
    path_tracer.set_depth(DEPTH);
    path_tracer.set_flip_y(true);
    path_tracer.set_gamma(2.2);
    for i in [1, 4, 20, 75] {
        let timer = util::Timer::start();
        path_tracer.set_iterations(i);
        path_tracer.render();
        timer.end(i);
    }

    let mut floats = vec![];
    floats.resize(path_tracer.get_target_buffer_len() as usize, 0.0);
    path_tracer
        .get_target_buffer(floats.as_mut_slice())
        .await
        .unwrap();
    let bytes = floats
        .iter()
        .map(|f| f32::clamp(f * 255.0, 0.0, 255.0) as u8)
        .collect();

    let image_buffer =
        image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_vec(WIDTH, HEIGHT, bytes).unwrap();
    image_buffer.save("target/image.png").unwrap();
}
