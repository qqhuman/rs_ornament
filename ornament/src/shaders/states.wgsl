struct DynamicState {
    current_iteration : f32,
}

struct ConstantState {
    depth : u32,
    width: u32,
    height: u32,
    flip_y: u32,
    inverted_gamma: f32
}