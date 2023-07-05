struct BvhNode {
    left_aabb_min_or_v0: vec3<f32>,
    left_or_custom_id: u32, // internal left node id / mesh id / triangle id, sphere id
    left_aabb_max_or_v1: vec3<f32>,
    right_or_material_index: u32,
    right_aabb_min_or_v2: vec3<f32>,
    node_type: u32, // 0 internal node, 1 mesh, 2 triangle, 3 sphere
    right_aabb_max_or_v3: vec3<f32>,
    // inverse transform: transform_id * 2
    // model transform:  transform_id * 2 + 1
    transform_id: u32, 
}

const finished_bottom_level_of_bvh: u32 = 0xffffffffu;
const max_bvh_depth = 64;
var<private> node_stack: array<u32, max_bvh_depth>;

fn bvh_hit(not_transformed_ray: Ray, hit: ptr<function, HitRecord>) -> bool {
    let t_min = 0.001;
    var t_max = 100000.0;
    
    let num_nodes = arrayLength(&bvh_nodes);
    var stack_top: i32 = 0;
    var addr = num_nodes - 1u;
    node_stack[stack_top] = addr;

    var hit_anything = false;
    var closest_so_far = t_max;

    var ray = not_transformed_ray;
    var invdir = safe_invdir(ray.direction);
    var oxinvdir = -ray.origin * invdir;

    let not_transformed_invdir = invdir;
    let not_transformed_oxinvdir = oxinvdir;
    var material_index: u32;
    var inverted_transform_id: u32;
    while stack_top >= 0 {
        let node = bvh_nodes[addr];
        switch node.node_type {
            // internal node
            case 0u: {
                let left = aabb_hit(node.left_aabb_min_or_v0, node.left_aabb_max_or_v1, invdir, oxinvdir, t_min, closest_so_far);
                let right = aabb_hit(node.right_aabb_min_or_v2, node.right_aabb_max_or_v3, invdir, oxinvdir, t_min, closest_so_far);
                let traverse_left = left.x <= left.y;
                let traverse_right = right.x <= right.y;
                
                if traverse_left {
                    stack_top++;
                    node_stack[stack_top] = node.left_or_custom_id;
                }

                if traverse_right {
                    stack_top++;
                    node_stack[stack_top] = node.right_or_material_index;
                }
            }
            // sphere
            case 1u: {
                let transform_id = node.transform_id * 2u;
                let transformed_ray = transform_ray(transform_id, ray);
                let t = sphere_hit(transformed_ray, t_min, closest_so_far);
                if t < closest_so_far {
                    closest_so_far = t;
                    hit_anything = true;

                    (*hit).t = t;
                    (*hit).p = ray_at(ray, t);
                    (*hit).material_index = node.right_or_material_index;
                    // outward_normal = hit - center
                    let outward_normal = (*hit).p - transform_point(transform_id + 1u, vec3<f32>(0.0));
                    hit_record_set_face_normal(hit, transformed_ray, outward_normal);
                }
            }
            // mesh
            case 2u, 3u: {
                // push signal to restore transformation after finshing mesh bvh
                stack_top++;
                node_stack[stack_top] = finished_bottom_level_of_bvh;

                // push mesh bvh
                stack_top++;
                node_stack[stack_top] = node.left_or_custom_id;

                inverted_transform_id = node.transform_id * 2u;
                material_index = node.right_or_material_index;
                ray = transform_ray(inverted_transform_id, ray);
                invdir = safe_invdir(ray.direction);
                oxinvdir = -ray.origin * invdir;
            }
            // triangle
            case 4u, default: {
                var uv: vec2<f32>;
                let t = triangle_hit(
                    ray, 
                    node.left_aabb_min_or_v0,
                    node.left_aabb_max_or_v1,
                    node.right_aabb_min_or_v2,
                    t_min, 
                    closest_so_far,
                    &uv
                );

                if t < closest_so_far {
                    closest_so_far = t;
                    hit_anything = true;

                    let tri = node.left_or_custom_id * 3u;
                    let n0 = normals[normal_indices[tri]];
                    let n1 = normals[normal_indices[tri + 1u]];
                    let n2 = normals[normal_indices[tri + 2u]];

                    let w = 1.0 - uv.x - uv.y;
                    let normal = uv.x * n1 + uv.y * n2 + w * n0;

                    (*hit).t = t;
                    (*hit).p = ray_at(not_transformed_ray, t);
                    let outward_normal = transform_normal(inverted_transform_id, normal);
                    hit_record_set_face_normal(hit, not_transformed_ray, outward_normal);
                    (*hit).material_index = material_index;
                }
            }
        }

        addr = node_stack[stack_top];
        stack_top--;

        if addr == finished_bottom_level_of_bvh {
            ray = not_transformed_ray;
            invdir = not_transformed_invdir;
            oxinvdir = not_transformed_oxinvdir;
            addr = node_stack[stack_top];
            stack_top--;
        }
    }

    return hit_anything;
}

fn mycopysign(a: f32, b: f32) -> f32 {
    if b < 0.0 {
        return -a;
    } else {
        return a;
    }
}

fn safe_invdir(d: vec3<f32>) -> vec3<f32> {
    let dirx = d.x;
    let diry = d.y;
    let dirz = d.z;
    let ooeps = 1e-5;
    var x: f32;
    if abs(dirx) > ooeps { 
        x = dirx; 
    } else { 
        x = mycopysign(ooeps, dirx); 
    };

    var y: f32;
    if abs(diry) > ooeps {
        y = diry;
    } else {
        y = mycopysign(ooeps, diry);
    }

    var z: f32;
    if abs(dirz) > ooeps {
        z = dirz;
    } else {
        z = mycopysign(ooeps, dirz);
    }

    return vec3<f32>(1.0 / x, 1.0 / y, 1.0 / z);
}

fn min3(val: vec3<f32>) -> f32 {
    return min(min(val.x, val.y), val.z);
}

fn max3(val: vec3<f32>) -> f32 {
    return max(max(val.x, val.y), val.z);
}

fn aabb_hit(mmin: vec3<f32>, mmax: vec3<f32>, invdir: vec3<f32>, oxinvdir: vec3<f32>, t_min: f32, t_max: f32) -> vec2<f32> {
    let f = mmax * invdir + oxinvdir;
    let n = mmin * invdir + oxinvdir;
    let tmax = max(f, n);
    let tmin = min(f, n);
    let max_t = min(min3(tmax), t_max);
    let min_t = max(max3(tmin), t_min);
    return vec2<f32>(min_t, max_t);
}

const epsilon: f32 = 1e-8;
fn triangle_hit(r: Ray, v1: vec3<f32>, v2: vec3<f32>, v3: vec3<f32>, t_min: f32, t_max: f32, uv: ptr<function, vec2<f32>>) -> f32 {
    let e1 = v2 - v1;
    let e2 = v3 - v1;

    let s1 = cross(r.direction, e2);
    let determinant = dot(s1, e1);
    if determinant < epsilon {
        return t_max;
    }

    let invd = 1.0 / determinant;

    let d = r.origin - v1;
    let u = dot(d, s1) * invd;

    // Barycentric coordinate U is outside range
    if u < 0.0 || u > 1.0 {
        return t_max;
    }

    let s2 = cross(d, e1);
    let v = dot(r.direction, s2) * invd;

    // Barycentric coordinate V is outside range
    if v < 0.0 || u + v > 1.0 {
        return t_max;
    }

    // t
    let t = dot(e2, s2) * invd;
    if t < t_min || t > t_max {
        return t_max;
    } else {
        *uv = vec2<f32>(u, v);
        return t;
    }
}