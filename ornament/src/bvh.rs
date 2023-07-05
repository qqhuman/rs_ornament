use std::{cell::Ref, cmp::Ordering, collections::HashMap};

use cgmath::Transform;
use rand::Rng;

use crate::{
    gpu_structs,
    math::{point3_max, point3_min, Aabb},
    Material, Mesh, MeshInstance, RcCell, Scene, Sphere,
};

#[derive(Default)]
pub struct Tree {
    pub nodes: Vec<Node>,
    pub normals: Vec<gpu_structs::Vector4>,
    pub normal_indices: Vec<u32>,
    pub transforms: Vec<gpu_structs::Transform>,
    pub materials: Vec<gpu_structs::Material>,
}

enum Leaf<'a> {
    Sphere(RcCell<Sphere>),
    Mesh(RcCell<Mesh>),
    MeshInstance(RcCell<MeshInstance>),
    Triangle(&'a Triangle),
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Node {
    left_aabb_min_or_v0: gpu_structs::Point3,
    left_or_custom_id: u32, // bvh node/top of mesh bvh/triangle id
    left_aabb_max_or_v1: gpu_structs::Point3,
    right_or_material_index: u32,
    right_aabb_min_or_v2: gpu_structs::Point3,
    node_type: u32, // 0 internal node, 1 sphere, 2 mesh, 3 triangle
    right_aabb_max_or_v3: gpu_structs::Point3,
    transform_id: u32,
}

pub fn build<'a>(scene: &Scene) -> Tree {
    let mut leafs = vec![];
    leafs.extend(scene.spheres.values().map(|s| Leaf::Sphere(s.clone())));
    leafs.extend(scene.meshes.values().map(|m| Leaf::Mesh(m.clone())));
    leafs.extend(
        scene
            .mesh_instances
            .values()
            .map(|mi| Leaf::MeshInstance(mi.clone())),
    );

    let mut bvh_tree = Tree::default();
    let mut instances_to_resolve = HashMap::new();
    let root = build_bvh_recursive(&mut bvh_tree, &mut leafs, &mut instances_to_resolve);
    bvh_tree.nodes.push(root);

    for (bvh_id, instance) in instances_to_resolve {
        bvh_tree.nodes[bvh_id as usize].left_or_custom_id = instance.borrow().mesh.borrow().bvh_id;
    }
    bvh_tree
}

fn build_bvh_recursive<'a>(
    bvh_tree: &mut Tree,
    shapes: &mut [Leaf],
    instances_to_resolve: &mut HashMap<u32, RcCell<MeshInstance>>,
) -> Node {
    let num_shapes = shapes.len() as u32;

    let node = match shapes.split_first_mut() {
        None => panic!("don't support empty bvh"),
        Some((head, [])) => match head {
            Leaf::Mesh(mesh) => {
                let mesh_top = from_mesh(bvh_tree, mesh.clone(), instances_to_resolve);

                bvh_tree.nodes.push(mesh_top);
                let mesh_top_id = bvh_tree.nodes.len() - 1;
                let mut mesh = mesh.borrow_mut();
                mesh.bvh_id = mesh_top_id as u32;

                let transform = mesh.transform;
                let inverse_transform = transform.inverse_transform().unwrap();
                bvh_tree.transforms.push(inverse_transform.into());
                bvh_tree.transforms.push(transform.into());
                let transform_id = bvh_tree.transforms.len() / 2 - 1;
                Node {
                    left_aabb_min_or_v0: mesh.aabb.min.into(),
                    left_aabb_max_or_v1: mesh.aabb.max.into(),
                    left_or_custom_id: mesh_top_id as u32,
                    right_or_material_index: get_material_index(bvh_tree, mesh.material.borrow()),
                    node_type: mesh.bvh_node_type(),
                    transform_id: transform_id as u32,

                    ..Default::default()
                }
            }
            Leaf::MeshInstance(instance) => {
                instances_to_resolve.insert(bvh_tree.nodes.len() as u32, instance.clone());
                let instance = instance.borrow();
                let transform = instance.transform;
                let inverse_transform = transform.inverse_transform().unwrap();
                bvh_tree.transforms.push(inverse_transform.into());
                bvh_tree.transforms.push(transform.into());
                let transform_id = bvh_tree.transforms.len() / 2 - 1;
                Node {
                    left_aabb_min_or_v0: instance.aabb.min.into(),
                    left_aabb_max_or_v1: instance.aabb.max.into(),
                    right_or_material_index: get_material_index(
                        bvh_tree,
                        instance.material.borrow(),
                    ),
                    node_type: instance.bvh_node_type(),
                    transform_id: transform_id as u32,

                    ..Default::default()
                }
            }
            Leaf::Sphere(sphere) => {
                let sphere = sphere.borrow();
                let transform = sphere.transform;
                let inverse_transform = transform.inverse_transform().unwrap();
                bvh_tree.transforms.push(inverse_transform.into());
                bvh_tree.transforms.push(transform.into());
                let transform_id = bvh_tree.transforms.len() / 2 - 1;
                Node {
                    left_aabb_min_or_v0: sphere.aabb.min.into(),
                    left_aabb_max_or_v1: sphere.aabb.max.into(),
                    right_or_material_index: get_material_index(bvh_tree, sphere.material.borrow()),
                    node_type: sphere.bvh_node_type(),
                    transform_id: transform_id as u32,

                    ..Default::default()
                }
            }
            Leaf::Triangle(triangle) => Node {
                left_aabb_min_or_v0: triangle.v0.into(),
                left_aabb_max_or_v1: triangle.v1.into(),
                right_aabb_min_or_v2: triangle.v2.into(),
                left_or_custom_id: triangle.triangle_index,
                node_type: triangle.bvh_node_type(),

                ..Default::default()
            },
        },
        _ => {
            let axis: i32 = rand::thread_rng().gen_range(0..=2);
            let comparator_ordering = |a: &Leaf, b: &Leaf| {
                if box_compare(a, b, axis) {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            };
            // Sort shapes based on the split axis
            shapes.sort_by(comparator_ordering);

            // Partition shapes into left and right subsets
            let mid = num_shapes / 2;
            let (left_shapes, right_shapes) = shapes.split_at_mut(mid as usize);

            // Recursively build BVH for left and right subsets
            let left = build_bvh_recursive(bvh_tree, left_shapes, instances_to_resolve);
            bvh_tree.nodes.push(left);
            let left = bvh_tree.nodes.len() as u32 - 1;

            let right = build_bvh_recursive(bvh_tree, right_shapes, instances_to_resolve);
            bvh_tree.nodes.push(right);
            let right = bvh_tree.nodes.len() as u32 - 1;

            let left_aabb = calculate_bounding_box(&left_shapes);
            let right_aabb = calculate_bounding_box(&right_shapes);

            Node {
                left_aabb_min_or_v0: left_aabb.min.into(),
                left_or_custom_id: left,
                left_aabb_max_or_v1: left_aabb.max.into(),
                right_or_material_index: right,
                right_aabb_min_or_v2: right_aabb.min.into(),
                node_type: NodeType::InternalNode as u32,
                right_aabb_max_or_v3: right_aabb.max.into(),
                ..Default::default()
            }
        }
    };

    node
}

fn get_material_index(bvh_tree: &mut Tree, material: Ref<Material>) -> u32 {
    match material.material_index {
        Some(id) => id,
        None => {
            bvh_tree
                .materials
                .push(gpu_structs::Material::from(material));
            return bvh_tree.materials.len() as u32 - 1;
        }
    }
}

fn from_mesh(
    bvh_tree: &mut Tree,
    mesh: RcCell<Mesh>,
    instances_to_resolve: &mut HashMap<u32, RcCell<MeshInstance>>,
) -> Node {
    let mesh = mesh.borrow_mut();
    let mut triangles = vec![];
    for mesh_triangle_index in 0..mesh.vertex_indices.len() / 3 {
        let v0 = mesh.vertices[mesh.vertex_indices[mesh_triangle_index * 3] as usize];
        let v1 = mesh.vertices[mesh.vertex_indices[mesh_triangle_index * 3 + 1] as usize];
        let v2 = mesh.vertices[mesh.vertex_indices[mesh_triangle_index * 3 + 2] as usize];
        let aabb = Aabb::new(
            point3_min(point3_min(v0, v1), v2),
            point3_max(point3_max(v0, v1), v2),
        );
        let global_triangle_index = bvh_tree.normal_indices.len() / 3 + mesh_triangle_index;
        triangles.push(Triangle {
            v0,
            v1,
            v2,
            triangle_index: global_triangle_index as u32,
            aabb,
        });
    }

    let mut leafs = vec![];
    leafs.extend(triangles.iter().map(|t| Leaf::Triangle(t)));

    bvh_tree.normal_indices.extend(
        mesh.normal_indices
            .iter()
            .map(|ni| ni + bvh_tree.normals.len() as u32),
    );
    bvh_tree
        .normals
        .extend(mesh.normals.iter().map(|n| [n.x, n.y, n.z, 0.0]));
    build_bvh_recursive(bvh_tree, &mut leafs, instances_to_resolve)
}

fn calculate_bounding_box(shapes: &[Leaf]) -> Aabb {
    let mut min = cgmath::Point3 {
        x: f32::MAX,
        y: f32::MAX,
        z: f32::MAX,
    };
    let mut max = cgmath::Point3 {
        x: f32::MIN,
        y: f32::MIN,
        z: f32::MIN,
    };

    for s in shapes {
        let aabb = match s {
            Leaf::Triangle(triangle) => triangle.aabb,
            Leaf::Sphere(sphere) => sphere.borrow().aabb,
            Leaf::Mesh(mesh) => mesh.borrow().aabb,
            Leaf::MeshInstance(instance) => instance.borrow().aabb,
        };
        min = point3_min(min, aabb.min);
        max = point3_max(max, aabb.max);
    }

    Aabb { min, max }
}

fn box_compare(a: &Leaf, b: &Leaf, axis: i32) -> bool {
    let box_a = match a {
        Leaf::Triangle(triangle) => triangle.aabb,
        Leaf::Sphere(sphere) => sphere.borrow().aabb,
        Leaf::Mesh(mesh) => mesh.borrow().aabb,
        Leaf::MeshInstance(instance) => instance.borrow().aabb,
    };
    let box_b = match b {
        Leaf::Triangle(triangle) => triangle.aabb,
        Leaf::Sphere(sphere) => sphere.borrow().aabb,
        Leaf::Mesh(mesh) => mesh.borrow().aabb,
        Leaf::MeshInstance(instance) => instance.borrow().aabb,
    };

    match axis {
        0 => box_a.min.x < box_b.min.x,
        1 => box_a.min.y < box_b.min.y,
        _ => box_a.min.z < box_b.min.z,
    }
}

struct Triangle {
    v0: cgmath::Point3<f32>,
    v1: cgmath::Point3<f32>,
    v2: cgmath::Point3<f32>,
    triangle_index: u32,
    aabb: Aabb,
}

enum NodeType {
    InternalNode = 0,
    Sphere = 1,
    Mesh = 2,
    MeshInstance = 3,
    Triangle = 4,
}

impl Triangle {
    fn bvh_node_type(&self) -> u32 {
        NodeType::Triangle as u32
    }
}

impl Sphere {
    fn bvh_node_type(&self) -> u32 {
        NodeType::Sphere as u32
    }
}

impl Mesh {
    fn bvh_node_type(&self) -> u32 {
        NodeType::Mesh as u32
    }
}

impl MeshInstance {
    fn bvh_node_type(&self) -> u32 {
        NodeType::MeshInstance as u32
    }
}
