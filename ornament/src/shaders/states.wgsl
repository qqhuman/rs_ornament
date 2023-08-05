struct DynamicState {
    current_iteration : f32,
    iterations: u32,
    reset_accumulation_buf: u32
}

struct ConstantState {
    depth : u32,
    width: u32,
    height: u32,
    flip_y: u32,
    inverted_gamma: f32
}