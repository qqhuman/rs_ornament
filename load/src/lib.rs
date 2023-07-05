use cgmath::{Array, InnerSpace, Transform};

use ornament::{
    Material, Mesh, RcCell,
};

pub fn point3_min(a: cgmath::Point3<f32>, b: cgmath::Point3<f32>) -> cgmath::Point3<f32> {
    cgmath::Point3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
}

pub fn point3_max(a: cgmath::Point3<f32>, b: cgmath::Point3<f32>) -> cgmath::Point3<f32> {
    cgmath::Point3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
}

#[cfg(target_os = "windows")]
pub fn assimp_mesh(
    path: &str,
    transform: cgmath::Matrix4<f32>,
    material: RcCell<Material>,
) -> Mesh {
    let scene = russimp::scene::Scene::from_file(
        path,
        vec![
            russimp::scene::PostProcess::Triangulate,
            russimp::scene::PostProcess::JoinIdenticalVertices,
            russimp::scene::PostProcess::SortByPrimitiveType,
            //russimp::scene::PostProcess::ForceGenerateNormals,
            russimp::scene::PostProcess::GenerateSmoothNormals,
        ],
    )
    .unwrap();

    if scene.meshes.len() > 1 {
        panic!("support only single model loading")
    }

    let mesh = &scene.meshes[0];

    let mut vertices = vec![];
    let mut normals = vec![];
    let vertex_indices = mesh
        .faces
        .iter()
        .map(|f| f.0.clone())
        .flatten()
        .collect::<Vec<u32>>();
    let normal_indices = vertex_indices.clone();

    for i in 0..mesh.vertices.len() {
        let v = mesh.vertices[i];
        vertices.push(cgmath::Point3::new(v.x, v.y, v.z));
    }

    for i in 0..mesh.normals.len() {
        let n = mesh.normals[i];
        normals.push(cgmath::Vector3::new(n.x, n.y, n.z).normalize());
    }

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

    for mesh_triangle_index in 0..vertex_indices.len() / 3 {
        let v0 = vertices[vertex_indices[mesh_triangle_index * 3] as usize];
        let v1 = vertices[vertex_indices[mesh_triangle_index * 3 + 1] as usize];
        let v2 = vertices[vertex_indices[mesh_triangle_index * 3 + 2] as usize];

        min = point3_min(min, point3_min(point3_min(v0, v1), v2));
        max = point3_max(max, point3_max(point3_max(v0, v1), v2));
    }

    let t = (min - cgmath::Point3::from_value(0.0)) + (max - min) * 0.5;
    let translate = cgmath::Matrix4::from_translation(t)
        .inverse_transform()
        .unwrap();
    let scale = cgmath::Matrix4::from_scale(1.0 / (max.y - min.y));
    let normalize_matrix = scale * translate;
    for v in vertices.iter_mut() {
        *v = normalize_matrix.transform_point(*v);
    }

    Mesh::new(
        vertices,
        vertex_indices,
        normals,
        normal_indices,
        transform,
        material,
    )
}
