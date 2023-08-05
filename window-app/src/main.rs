mod app;
mod controllers;

fn main() {
    pollster::block_on(app::run());
}
