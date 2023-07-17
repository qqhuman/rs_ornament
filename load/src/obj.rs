use cgmath::InnerSpace;
use std::{fs::File, io::BufReader};

use ornament::{Material, Mesh, RcCell};

use crate::{normalize_vertices, Error};

pub fn mesh(
    path: &str,
    transform: cgmath::Matrix4<f32>,
    material: RcCell<Material>,
) -> Result<Mesh, Error> {
    let input = BufReader::new(File::open(path).map_err(|_| Error::OpenFile)?);
    let model: obj::Obj<obj::Vertex, u32> = obj::load_obj(input).map_err(|_| Error::Obj)?;

    let mut vertices = vec![];
    let mut normals = vec![];
    let vertex_indices = model.indices.clone();
    let normal_indices = model.indices.clone();
    for vertex in model.vertices.iter() {
        let v: [f32; 3] = vertex.position;
        let n: [f32; 3] = vertex.normal;
        vertices.push(cgmath::Point3::new(v[0], v[1], v[2]));
        normals.push(cgmath::Vector3::new(n[0], n[1], n[2]).normalize());
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
