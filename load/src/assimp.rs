use cgmath::InnerSpace;
use ornament::{Material, Mesh, RcCell};

use crate::{normalize_vertices, Error};

pub fn mesh(
    path: &str,
    transform: cgmath::Matrix4<f32>,
    material: RcCell<Material>,
) -> Result<Mesh, Error> {
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
    .map_err(|_| Error::Assimp)?;

    let mesh = &scene.meshes.get(0).ok_or(Error::MeshesNotFound)?;

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

    normalize_vertices(&mut vertices, &vertex_indices)?;

    Ok(Mesh::new(
        vertices,
        vertex_indices,
        normals,
        normal_indices,
        transform,
        material,
    ))
}
