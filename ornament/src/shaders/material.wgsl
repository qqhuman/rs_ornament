struct Material {
    albedo: vec3<f32>,
    fuzz: f32,
    ior: f32,
    material_type: u32
}

fn lambertian_scatter(material: Material, r_in: Ray, rec: HitRecord, attenuation: ptr<function, vec3<f32>>, scattered: ptr<function, Ray>) -> bool {
    var scatter_direction = rec.normal + random_unit_vector();

    // Catch degenerate scatter direction
    if near_zero(scatter_direction) {
        scatter_direction = rec.normal;
    }

    (*scattered) = Ray(rec.p, scatter_direction);
    (*attenuation) = material.albedo;
    return true;
}

fn metal_scatter(material: Material, r_in: Ray, rec: HitRecord, attenuation: ptr<function, vec3<f32>>, scattered: ptr<function, Ray>) -> bool {
    let scattered_direction = reflect(normalize(r_in.direction), rec.normal) + material.fuzz * random_in_unit_sphere();
    (*scattered) = Ray(rec.p, scattered_direction);
    (*attenuation) = material.albedo;
    return (dot(scattered_direction, rec.normal) > 0.0);
}

fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    // Use Schlick's approximation for reflectance.
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

fn dielectric_scatter(material: Material, r_in: Ray, rec: HitRecord, attenuation: ptr<function, vec3<f32>>, scattered: ptr<function, Ray>) -> bool {
    (*attenuation) = vec3<f32>(1.0);
    var refraction_ratio = material.ior;
    if rec.front_face {
        refraction_ratio = 1.0 / material.ior;
    }

    let unit_direction = normalize(r_in.direction);
    let cos_theta = min(dot(-unit_direction, rec.normal), 1.0);
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    let cannot_refract = refraction_ratio * sin_theta > 1.0;
    var direction: vec3<f32>;

    if cannot_refract || reflectance(cos_theta, refraction_ratio) > random_f32() {
        direction = reflect(unit_direction, rec.normal);
    }
    else {
        direction = refract(unit_direction, rec.normal, refraction_ratio);
    }

    (*scattered) = Ray(rec.p, direction);
    return true;
}

fn material_scatter(r_in: Ray, rec: HitRecord, attenuation: ptr<function, vec3<f32>>, scattered: ptr<function, Ray>) -> bool {
    let material = materials[rec.material_index];
    switch material.material_type {
        case 0u {
            return lambertian_scatter(material, r_in, rec, attenuation, scattered);
        }
        case 1u {
            return metal_scatter(material, r_in, rec, attenuation, scattered);
        }
        case 2u {
            return dielectric_scatter(material, r_in, rec, attenuation, scattered);
        }
        default {
            return false;
        }
    }
}