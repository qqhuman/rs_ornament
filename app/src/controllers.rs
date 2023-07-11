use winit::{event::{ElementState, KeyEvent, WindowEvent}, keyboard::KeyCode};

pub struct Camera {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    dirty: bool,
}

impl Camera {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            dirty: true,
        }
    }

    pub fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        physical_key,
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match physical_key {
                    KeyCode::KeyW | KeyCode::ArrowUp => {
                        self.is_forward_pressed = is_pressed;
                        self.dirty = true;
                    }
                    KeyCode::KeyA | KeyCode::ArrowLeft => {
                        self.is_left_pressed = is_pressed;
                        self.dirty = true;
                    }
                    KeyCode::KeyS | KeyCode::ArrowDown => {
                        self.is_backward_pressed = is_pressed;
                        self.dirty = true;
                    }
                    KeyCode::KeyD | KeyCode::ArrowRight => {
                        self.is_right_pressed = is_pressed;
                        self.dirty = true;
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        self.dirty
    }

    pub fn update_camera(&mut self, camera: &mut ornament::Camera) {
        if !self.dirty {
            return;
        }

        use cgmath::InnerSpace;
        let target = camera.get_look_at();
        let mut eye = camera.get_look_from();
        let up = camera.get_vup();
        let forward = target - eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        // Prevents glitching when camera gets too close to the
        // center of the scene.
        if self.is_forward_pressed && forward_mag > self.speed {
            eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            eye -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(up);

        // Redo radius calc in case the fowrard/backward is pressed.
        let forward = target - eye;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            // Rescale the distance between the target and eye so
            // that it doesn't change. The eye therefore still
            // lies on the circle made by the target and eye.
            eye = target - (forward - right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            eye = target - (forward + right * self.speed).normalize() * forward_mag;
        }

        camera.set_look_at(eye, target, up);
        self.dirty = false;
    }
}
