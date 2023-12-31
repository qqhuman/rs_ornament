use cgmath::EuclideanSpace;
use rand::{rngs::StdRng, Rng, SeedableRng};

use ornament::{Camera, Color, Material, Mesh, MeshInstance, Scene, Sphere};

fn random_color(rng: &mut StdRng) -> Color {
    Color::new(rng.gen(), rng.gen(), rng.gen())
}

fn random_color_between(rng: &mut StdRng, min: f32, max: f32) -> Color {
    Color::new(
        rng.gen_range(min..max),
        rng.gen_range(min..max),
        rng.gen_range(min..max),
    )
}

fn rng() -> StdRng {
    StdRng::seed_from_u64(45)
}

pub fn length(v: cgmath::Vector3<f32>) -> f32 {
    f32::sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
}

pub fn random_scene_spheres(aspect_ratio: f32) -> Scene {
    let vfov = 20.0;
    let lookfrom = cgmath::Point3::new(13.0, 2.0, 3.0);
    let lookat = cgmath::Point3::new(0.0, 0.0, 0.0);
    let vup = cgmath::Vector3::new(0.0, 1.0, 0.0);
    let aperture = 0.1;
    let focus_dist = 10.0;
    let camera = Camera::new(
        lookfrom,
        lookat,
        vup,
        aspect_ratio,
        vfov,
        aperture,
        focus_dist,
    );

    let mut scene = Scene::new(camera);
    let mut rng = rng();

    scene.add_sphere(Sphere::new(
        cgmath::Point3::new(0.0, -1000.0, 0.0),
        1000.0,
        Material::lambertian(Color::new(0.5, 0.5, 0.5)),
    ));

    for a in -11..11 {
        for b in -11..11 {
            let a = a as f32;
            let b = b as f32;
            let choose_mat: f32 = rng.gen();
            let center =
                cgmath::Point3::new(a + 0.9 * rng.gen::<f32>(), 0.2, b + 0.9 * rng.gen::<f32>());

            if length(center - cgmath::Point3::new(4.0, 0.2, 0.0)) > 0.9 {
                let material = if choose_mat < 0.8 {
                    let color1 = random_color(&mut rng);
                    let color2 = random_color(&mut rng);
                    let albedo = Color::new(
                        color1.x * color2.x,
                        color1.y * color2.y,
                        color1.z * color2.z,
                    );
                    Material::lambertian(albedo)
                } else if choose_mat < 0.95 {
                    let albedo = random_color_between(&mut rng, 0.5, 1.0);
                    let fuzz = rng.gen_range(0.0..0.5);
                    Material::metal(albedo, fuzz)
                } else {
                    Material::dielectric(1.5)
                };
                scene.add_sphere(Sphere::new(center, 0.2, material));
            }
        }
    }

    scene.add_sphere(Sphere::new(
        cgmath::Point3::new(0.0, 1.0, 0.0),
        1.0,
        Material::dielectric(1.5),
    ));

    scene.add_sphere(Sphere::new(
        cgmath::Point3::new(-4.0, 1.0, 0.0),
        1.0,
        Material::lambertian(Color::new(0.4, 0.2, 0.1)),
    ));

    scene.add_sphere(Sphere::new(
        cgmath::Point3::new(4.0, 1.0, 0.0),
        1.0,
        Material::metal(Color::new(0.7, 0.6, 0.5), 0.0),
    ));

    scene
}

pub fn random_scene_mix_meshes_and_spheres(aspect_ratio: f32) -> Scene {
    let vfov = 20.0;
    let lookfrom = cgmath::Point3::new(13.0, 2.0, 3.0);
    let lookat = cgmath::Point3::new(0.0, 0.0, 0.0);
    let vup = cgmath::Vector3::new(0.0, 1.0, 0.0);
    let aperture = 0.1;
    let focus_dist = 10.0;
    let camera = Camera::new(
        lookfrom,
        lookat,
        vup,
        aspect_ratio,
        vfov,
        aperture,
        focus_dist,
    );

    let mut scene = Scene::new(camera);
    let mut rng = rng();

    scene.add_sphere(Sphere::new(
        cgmath::Point3::new(0.0, -1000.0, 0.0),
        1000.0,
        Material::lambertian(Color::new(0.5, 0.5, 0.5)),
    ));

    let mesh_sphere = scene.add_mesh(Mesh::sphere(
        cgmath::Point3::new(0.0, 1.0, 0.0),
        1.0,
        Material::dielectric(1.5),
    ));

    scene.add_mesh_instances(MeshInstance::new(
        mesh_sphere.clone(),
        Material::lambertian(Color::new(0.4, 0.2, 0.1)),
        cgmath::Matrix4::from_translation(cgmath::Vector3::new(-4.0, 1.0, 0.0)),
    ));

    scene.add_mesh_instances(MeshInstance::new(
        mesh_sphere.clone(),
        Material::metal(Color::new(0.7, 0.6, 0.5), 0.0),
        cgmath::Matrix4::from_translation(cgmath::Vector3::new(4.0, 1.0, 0.0)),
    ));

    for a in -11..11 {
        for b in -11..11 {
            let a = a as f32;
            let b = b as f32;
            let choose_mat: f32 = rng.gen();
            let center =
                cgmath::Point3::new(a + 0.9 * rng.gen::<f32>(), 0.2, b + 0.9 * rng.gen::<f32>());

            if length(center - cgmath::Point3::new(4.0, 0.2, 0.0)) > 0.9 {
                let material = if choose_mat < 0.8 {
                    let color1 = random_color(&mut rng);
                    let color2 = random_color(&mut rng);
                    let albedo = Color::new(
                        color1.x * color2.x,
                        color1.y * color2.y,
                        color1.z * color2.z,
                    );
                    Material::lambertian(albedo)
                } else if choose_mat < 0.95 {
                    let albedo = random_color_between(&mut rng, 0.5, 1.0);
                    let fuzz = rng.gen_range(0.0..0.5);
                    Material::metal(albedo, fuzz)
                } else {
                    Material::dielectric(1.5)
                };
                scene.add_mesh_instances(MeshInstance::new(
                    mesh_sphere.clone(),
                    material,
                    cgmath::Matrix4::from_translation(center.to_vec())
                        * cgmath::Matrix4::from_scale(0.2),
                ));
            }
        }
    }

    scene
}

pub fn random_scene_meshes(aspect_ratio: f32) -> Scene {
    let vfov = 20.0;
    let lookfrom = cgmath::Point3::new(13.0, 2.0, 3.0);
    let lookat = cgmath::Point3::new(0.0, 0.0, 0.0);
    let vup = cgmath::Vector3::new(0.0, 1.0, 0.0);
    let aperture = 0.1;
    let focus_dist = 10.0;
    let camera = Camera::new(
        lookfrom,
        lookat,
        vup,
        aspect_ratio,
        vfov,
        aperture,
        focus_dist,
    );

    let mut scene = Scene::new(camera);
    let mut rng = rng();

    let mesh_sphere = scene.add_mesh(Mesh::sphere(
        cgmath::Point3::new(0.0, -1000.0, 0.0),
        1000.0,
        Material::lambertian(Color::new(0.5, 0.5, 0.5)),
    ));

    scene.add_mesh_instances(MeshInstance::new(
        mesh_sphere.clone(),
        Material::dielectric(1.5),
        cgmath::Matrix4::from_translation(cgmath::Vector3::new(0.0, 1.0, 0.0)),
    ));

    scene.add_mesh_instances(MeshInstance::new(
        mesh_sphere.clone(),
        Material::lambertian(Color::new(0.4, 0.2, 0.1)),
        cgmath::Matrix4::from_translation(cgmath::Vector3::new(-4.0, 1.0, 0.0)),
    ));

    scene.add_mesh_instances(MeshInstance::new(
        mesh_sphere.clone(),
        Material::metal(Color::new(0.7, 0.6, 0.5), 0.0),
        cgmath::Matrix4::from_translation(cgmath::Vector3::new(4.0, 1.0, 0.0)),
    ));

    for a in -11..11 {
        for b in -11..11 {
            let a = a as f32;
            let b = b as f32;
            let choose_mat: f32 = rng.gen();
            let center =
                cgmath::Point3::new(a + 0.9 * rng.gen::<f32>(), 0.2, b + 0.9 * rng.gen::<f32>());

            if length(center - cgmath::Point3::new(4.0, 0.2, 0.0)) > 0.9 {
                let material = if choose_mat < 0.8 {
                    let color1 = random_color(&mut rng);
                    let color2 = random_color(&mut rng);
                    let albedo = Color::new(
                        color1.x * color2.x,
                        color1.y * color2.y,
                        color1.z * color2.z,
                    );
                    Material::lambertian(albedo)
                } else if choose_mat < 0.95 {
                    let albedo = random_color_between(&mut rng, 0.5, 1.0);
                    let fuzz = rng.gen_range(0.0..0.5);
                    Material::metal(albedo, fuzz)
                } else {
                    Material::dielectric(1.5)
                };
                scene.add_mesh_instances(MeshInstance::new(
                    mesh_sphere.clone(),
                    material,
                    cgmath::Matrix4::from_translation(center.to_vec())
                        * cgmath::Matrix4::from_scale(0.2),
                ));
            }
        }
    }

    scene
}

pub fn random_scene_with_3_lucy(aspect_ratio: f32) -> Scene {
    let vfov = 20.0;
    let lookfrom = cgmath::Point3::new(13.0, 2.0, 3.0);
    let lookat = cgmath::Point3::new(0.0, 0.0, 0.0);
    let vup = cgmath::Vector3::new(0.0, 1.0, 0.0);
    let aperture = 0.0;
    let focus_dist = 10.0;
    let camera = Camera::new(
        lookfrom,
        lookat,
        vup,
        aspect_ratio,
        vfov,
        aperture,
        focus_dist,
    );

    let mut scene = Scene::new(camera);
    let mut rng = rng();

    scene.add_sphere(Sphere::new(
        cgmath::Point3::new(0.0, -1000.0, 0.0),
        1000.0,
        Material::lambertian(Color::new(0.5, 0.5, 0.5)),
    ));

    let base_lucy_transform =
        cgmath::Matrix4::from_angle_y(cgmath::Rad(std::f32::consts::PI / 2.0))
            * cgmath::Matrix4::from_scale(2.0);

    let lucy = scene.add_mesh(
        load::obj::mesh(
            "examples/models/lucy.obj",
            cgmath::Matrix4::from_translation(cgmath::Vector3::new(-4.0, 1.0, 0.0))
                * base_lucy_transform,
            Material::lambertian(Color::new(0.4, 0.2, 0.1)),
        )
        .unwrap(),
    );

    scene.add_mesh_instances(MeshInstance::new(
        lucy.clone(),
        Material::dielectric(1.5),
        cgmath::Matrix4::from_translation(cgmath::Vector3::new(0.0, 1.0, 0.0))
            * base_lucy_transform,
    ));

    scene.add_mesh_instances(MeshInstance::new(
        lucy.clone(),
        Material::metal(Color::new(0.7, 0.6, 0.5), 0.0),
        cgmath::Matrix4::from_translation(cgmath::Vector3::new(4.0, 1.0, 0.0))
            * base_lucy_transform,
    ));

    let mut mesh_sphere: Option<ornament::RcCell<Mesh>> = None;

    for a in -11..11 {
        for b in -11..11 {
            let a = a as f32;
            let b = b as f32;
            let choose_mat: f32 = rng.gen();
            let center =
                cgmath::Point3::new(a + 0.9 * rng.gen::<f32>(), 0.2, b + 0.9 * rng.gen::<f32>());

            if length(center - cgmath::Point3::new(4.0, 0.2, 0.0)) > 0.9 {
                let material = if choose_mat < 0.8 {
                    let color1 = random_color(&mut rng);
                    let color2 = random_color(&mut rng);
                    let albedo = Color::new(
                        color1.x * color2.x,
                        color1.y * color2.y,
                        color1.z * color2.z,
                    );
                    Material::lambertian(albedo)
                } else if choose_mat < 0.95 {
                    let albedo = random_color_between(&mut rng, 0.5, 1.0);
                    let fuzz = rng.gen_range(0.0..0.5);
                    Material::metal(albedo, fuzz)
                } else {
                    Material::dielectric(1.5)
                };
                match &mesh_sphere {
                    Some(mesh_sphere) => {
                        scene.add_mesh_instances(MeshInstance::new(
                            mesh_sphere.clone(),
                            material,
                            cgmath::Matrix4::from_translation(center.to_vec())
                                * cgmath::Matrix4::from_scale(0.2),
                        ));
                    }
                    None => {
                        mesh_sphere = Some(scene.add_mesh(Mesh::sphere(
                            center,
                            0.2,
                            Material::lambertian(Color::new(0.5, 0.5, 0.5)),
                        )));
                    }
                }
            }
        }
    }

    scene
}

#[cfg(target_os = "windows")]
pub fn random_scene_with_3_statuette(aspect_ratio: f32) -> Scene {
    let vfov = 20.0;
    let lookfrom = cgmath::Point3::new(13.0, 2.0, 3.0);
    let lookat = cgmath::Point3::new(0.0, 0.0, 0.0);
    let vup = cgmath::Vector3::new(0.0, 1.0, 0.0);
    let aperture = 0.05;
    let focus_dist = 10.0;
    let camera = Camera::new(
        lookfrom,
        lookat,
        vup,
        aspect_ratio,
        vfov,
        aperture,
        focus_dist,
    );

    let mut scene = Scene::new(camera);
    let mut rng = rng();

    scene.add_sphere(Sphere::new(
        cgmath::Point3::new(0.0, -1000.0, 0.0),
        1000.0,
        Material::lambertian(Color::new(0.5, 0.5, 0.5)),
    ));

    let base_statuette_transform = cgmath::Matrix4::from_scale(2.0);

    let statuette = scene.add_mesh(
        load::assimp::mesh(
            "examples/models/xyzrgb_statuette.ply",
            cgmath::Matrix4::from_translation(cgmath::Vector3::new(-4.0, 1.0, 0.0))
                * base_statuette_transform,
            Material::lambertian(Color::new(0.4, 0.2, 0.1)),
        )
        .unwrap(),
    );

    scene.add_mesh_instances(MeshInstance::new(
        statuette.clone(),
        Material::dielectric(1.5),
        cgmath::Matrix4::from_translation(cgmath::Vector3::new(0.0, 1.0, 0.0))
            * base_statuette_transform,
    ));

    scene.add_mesh_instances(MeshInstance::new(
        statuette.clone(),
        Material::metal(Color::new(0.7, 0.6, 0.5), 0.0),
        cgmath::Matrix4::from_translation(cgmath::Vector3::new(4.0, 1.0, 0.0))
            * base_statuette_transform,
    ));

    let mut mesh_sphere: Option<ornament::RcCell<Mesh>> = None;

    for a in -11..11 {
        for b in -11..11 {
            let a = a as f32;
            let b = b as f32;
            let choose_mat: f32 = rng.gen();
            let center =
                cgmath::Point3::new(a + 0.9 * rng.gen::<f32>(), 0.2, b + 0.9 * rng.gen::<f32>());

            if length(center - cgmath::Point3::new(4.0, 0.2, 0.0)) > 0.9 {
                let material = if choose_mat < 0.8 {
                    let color1 = random_color(&mut rng);
                    let color2 = random_color(&mut rng);
                    let albedo = Color::new(
                        color1.x * color2.x,
                        color1.y * color2.y,
                        color1.z * color2.z,
                    );
                    Material::lambertian(albedo)
                } else if choose_mat < 0.95 {
                    let albedo = random_color_between(&mut rng, 0.5, 1.0);
                    let fuzz = rng.gen_range(0.0..0.5);
                    Material::metal(albedo, fuzz)
                } else {
                    Material::dielectric(1.5)
                };
                match &mesh_sphere {
                    Some(mesh_sphere) => {
                        scene.add_mesh_instances(MeshInstance::new(
                            mesh_sphere.clone(),
                            material,
                            cgmath::Matrix4::from_translation(center.to_vec())
                                * cgmath::Matrix4::from_scale(0.2),
                        ));
                    }
                    None => {
                        mesh_sphere = Some(scene.add_mesh(Mesh::sphere(
                            center,
                            0.2,
                            Material::lambertian(Color::new(0.5, 0.5, 0.5)),
                        )));
                    }
                }
            }
        }
    }

    scene
}

fn quad_center_from_book(
    q: cgmath::Point3<f32>,
    u: cgmath::Vector3<f32>,
    v: cgmath::Vector3<f32>,
) -> cgmath::Point3<f32> {
    let u = u * 0.5;
    let v = v * 0.5;
    q + u + v
}

pub fn quads(aspect_ratio: f32) -> Scene {
    let vfov = 80.0;
    let lookfrom = cgmath::Point3::new(0.0, 0.0, 9.0);
    let lookat = cgmath::Point3::new(0.0, 0.0, 0.0);
    let vup = cgmath::Vector3::new(0.0, 1.0, 0.0);
    let aperture = 0.0;
    let focus_dist = 10.0;
    let camera = Camera::new(
        lookfrom,
        lookat,
        vup,
        aspect_ratio,
        vfov,
        aperture,
        focus_dist,
    );

    let mut scene = Scene::new(camera);

    let left_red = Material::lambertian(Color::new(1.0, 0.2, 0.2));
    let back_green = Material::lambertian(Color::new(0.2, 1.0, 0.2));
    let right_blue = Material::lambertian(Color::new(0.2, 0.2, 1.0));
    let upper_orange = Material::lambertian(Color::new(1.0, 0.5, 0.0));
    let lower_teal = Material::lambertian(Color::new(0.2, 0.8, 0.8));

    scene.add_mesh(Mesh::plane(
        quad_center_from_book(
            cgmath::Point3::new(-3.0, -2.0, 5.0),
            cgmath::Vector3::new(0.0, 0.0, -4.0),
            cgmath::Vector3::new(0.0, 4.0, 0.0),
        ),
        4.0,
        4.0,
        cgmath::Vector3::unit_x(),
        left_red,
    ));
    scene.add_mesh(Mesh::plane(
        quad_center_from_book(
            cgmath::Point3::new(-2.0, -2.0, 0.0),
            cgmath::Vector3::new(4.0, 0.0, 0.0),
            cgmath::Vector3::new(0.0, 4.0, 0.0),
        ),
        4.0,
        4.0,
        cgmath::Vector3::unit_z(),
        back_green,
    ));
    scene.add_mesh(Mesh::plane(
        quad_center_from_book(
            cgmath::Point3::new(3.0, -2.0, 1.0),
            cgmath::Vector3::new(0.0, 0.0, 4.0),
            cgmath::Vector3::new(0.0, 4.0, 0.0),
        ),
        4.0,
        4.0,
        -cgmath::Vector3::unit_x(),
        right_blue,
    ));
    scene.add_mesh(Mesh::plane(
        quad_center_from_book(
            cgmath::Point3::new(-2.0, 3.0, 1.0),
            cgmath::Vector3::new(4.0, 0.0, 0.0),
            cgmath::Vector3::new(0.0, 0.0, 4.0),
        ),
        4.0,
        4.0,
        -cgmath::Vector3::unit_y(),
        upper_orange,
    ));

    scene.add_mesh(Mesh::plane(
        quad_center_from_book(
            cgmath::Point3::new(-2.0, -3.0, 5.0),
            cgmath::Vector3::new(4.0, 0.0, 0.0),
            cgmath::Vector3::new(0.0, 0.0, -4.0),
        ),
        4.0,
        4.0,
        cgmath::Vector3::unit_y(),
        lower_teal,
    ));

    scene.add_mesh(Mesh::sphere(
        cgmath::Point3::new(0.0, 0.0, 3.0),
        1.0,
        Material::diffuse_light(Color::new(3.0, 3.0, 3.0)),
    ));

    scene
}

pub fn empty_cornell_box(aspect_ratio: f32) -> Scene {
    let vfov = 40.0;
    let lookfrom = cgmath::Point3::new(278.0, 278.0, -800.0);
    let lookat = cgmath::Point3::new(278.0, 278.0, 0.0);
    let vup = cgmath::Vector3::new(0.0, 1.0, 0.0);
    let aperture = 0.0;
    let focus_dist = 10.0;
    let camera = Camera::new(
        lookfrom,
        lookat,
        vup,
        aspect_ratio,
        vfov,
        aperture,
        focus_dist,
    );

    let mut scene = Scene::new(camera);

    let red = Material::lambertian(Color::new(0.65, 0.05, 0.05));
    let white = Material::lambertian(Color::new(0.73, 0.73, 0.73));
    let green = Material::lambertian(Color::new(0.12, 0.45, 0.15));
    let light = Material::diffuse_light(Color::new(15.0, 15.0, 15.0));

    scene.add_mesh(Mesh::plane(
        quad_center_from_book(
            cgmath::Point3::new(555.0, 0.0, 0.0),
            cgmath::Vector3::new(0.0, 555.0, 0.0),
            cgmath::Vector3::new(0.0, 0.0, 555.0),
        ),
        555.0,
        555.0,
        cgmath::Vector3::unit_x(),
        green,
    ));
    scene.add_mesh(Mesh::plane(
        quad_center_from_book(
            cgmath::Point3::new(0.0, 0.0, 0.0),
            cgmath::Vector3::new(0.0, 555.0, 0.0),
            cgmath::Vector3::new(0.0, 0.0, 555.0),
        ),
        555.0,
        555.0,
        -cgmath::Vector3::unit_x(),
        red,
    ));
    // ??????????????????????? not a square
    scene.add_mesh(Mesh::plane(
        quad_center_from_book(
            cgmath::Point3::new(343.0, 554.0, 332.0),
            cgmath::Vector3::new(-130.0, 0.0, 0.0),
            cgmath::Vector3::new(0.0, 0.0, -105.0),
        ),
        130.0,
        105.0,
        -cgmath::Vector3::unit_y(),
        light,
    ));
    scene.add_mesh(Mesh::plane(
        quad_center_from_book(
            cgmath::Point3::new(0.0, 0.0, 0.0),
            cgmath::Vector3::new(555.0, 0.0, 0.0),
            cgmath::Vector3::new(0.0, 0.0, 555.0),
        ),
        555.0,
        555.0,
        cgmath::Vector3::unit_y(),
        white.clone(),
    ));

    scene.add_mesh(Mesh::plane(
        quad_center_from_book(
            cgmath::Point3::new(555.0, 555.0, 555.0),
            cgmath::Vector3::new(-555.0, 0.0, 0.0),
            cgmath::Vector3::new(0.0, 0.0, -555.0),
        ),
        555.0,
        555.0,
        -cgmath::Vector3::unit_y(),
        white.clone(),
    ));

    scene.add_mesh(Mesh::plane(
        quad_center_from_book(
            cgmath::Point3::new(0.0, 0.0, 555.0),
            cgmath::Vector3::new(555.0, 0.0, 0.0),
            cgmath::Vector3::new(0.0, 555.0, 0.0),
        ),
        555.0,
        555.0,
        cgmath::Vector3::unit_z(),
        white,
    ));

    scene
}

pub fn cornell_box_with_lucy(aspect_ratio: f32) -> Scene {
    let mut scene = empty_cornell_box(aspect_ratio);

    let height = 400.0;
    let mesh = scene.add_mesh(
        load::obj::mesh(
            "examples/models/lucy.obj",
            cgmath::Matrix4::from_translation(cgmath::Vector3::new(
                265.0 * 1.5,
                height / 2.0,
                295.0,
            )) * cgmath::Matrix4::from_angle_y(cgmath::Rad(std::f32::consts::PI * 1.5))
                * cgmath::Matrix4::from_scale(height),
            Material::lambertian(Color::new(0.4, 0.2, 0.1)),
        )
        .unwrap(),
    );

    let height = 200.0;
    scene.add_mesh_instances(MeshInstance::new(
        mesh,
        Material::lambertian(Color::new(0.4, 0.2, 0.1)),
        cgmath::Matrix4::from_translation(cgmath::Vector3::new(130.0 * 1.5, height / 2.0, 65.0))
            * cgmath::Matrix4::from_angle_y(cgmath::Rad(std::f32::consts::PI * 1.25))
            * cgmath::Matrix4::from_scale(height),
    ));

    scene
}

#[cfg(target_os = "windows")]
pub fn cornell_box_with_statuette(aspect_ratio: f32) -> Scene {
    let mut scene = empty_cornell_box(aspect_ratio);

    let height = 400.0;
    let mesh = scene.add_mesh(
        load::assimp::mesh(
            "examples/models/xyzrgb_statuette.ply",
            cgmath::Matrix4::from_translation(cgmath::Vector3::new(
                265.0 * 1.5,
                height / 2.0,
                295.0,
            )) * cgmath::Matrix4::from_angle_y(cgmath::Rad(std::f32::consts::PI * 1.5))
                * cgmath::Matrix4::from_scale(height),
            Material::lambertian(Color::new(0.4, 0.2, 0.1)),
        )
        .unwrap(),
    );

    let height = 200.0;
    scene.add_mesh_instances(MeshInstance::new(
        mesh,
        Material::lambertian(Color::new(0.4, 0.2, 0.1)),
        cgmath::Matrix4::from_translation(cgmath::Vector3::new(130.0 * 1.5, height / 2.0, 65.0))
            * cgmath::Matrix4::from_angle_y(cgmath::Rad(std::f32::consts::PI * 1.25))
            * cgmath::Matrix4::from_scale(height),
    ));

    scene
}

pub fn dielectric_test(aspect_ratio: f32) -> Scene {
    let vfov = 90.0;
    let lookfrom = cgmath::Point3::new(0.0, 0.0, 0.0);
    let lookat = cgmath::Point3::new(0.0, 0.0, -1.0);
    let vup = cgmath::Vector3::new(0.0, 1.0, 0.0);
    let aperture = 0.0;
    let focus_dist = 10.0;
    let camera = Camera::new(
        lookfrom,
        lookat,
        vup,
        aspect_ratio,
        vfov,
        aperture,
        focus_dist,
    );

    let mut scene = Scene::new(camera);

    let materia_ground = Material::lambertian(Color::new(0.8, 0.8, 0.0));
    let material = Material::dielectric(1.5);

    let mesh_sphere = scene.add_mesh(Mesh::sphere(
        cgmath::Point3::new(0.0, -100.5, -1.0),
        100.0,
        materia_ground,
    ));

    scene.add_mesh_instances(MeshInstance::new(
        mesh_sphere.clone(),
        material.clone(),
        cgmath::Matrix4::from_translation(cgmath::Vector3::new(0.0, 0.0, -1.0))
            * cgmath::Matrix4::from_scale(0.5),
    ));

    scene.add_mesh_instances(MeshInstance::new(
        mesh_sphere.clone(),
        material.clone(),
        cgmath::Matrix4::from_translation(cgmath::Vector3::new(-1.0, 0.0, -1.0))
            * cgmath::Matrix4::from_scale(0.5),
    ));

    // scene.add_mesh_instances(MeshInstance::new(
    //     mesh_sphere.clone(),
    //     material_right,
    //     cgmath::Matrix4::from_translation(cgmath::Vector3::new(1.0, 0.0, -1.0))
    //         * cgmath::Matrix4::from_scale(0.5),
    // ));

    // scene.add_sphere(Sphere::new(cgmath::Point3::new(0.0, 0.0, -1.0), 0.5, material_center));
    //scene.add_sphere(Sphere::new(cgmath::Point3::new(-1.0, 0.0, -1.0), 0.5, material_left));
    scene.add_sphere(Sphere::new(
        cgmath::Point3::new(1.0, 0.0, -1.0),
        0.5,
        material,
    ));

    scene
}
