use std::{
    intrinsics::transmute,
    io::Read,
    mem::MaybeUninit,
};

use ssun_math::Float3;
use windows::Win32::Graphics::Direct3D12::ID3D12Device;

use crate::buffers::*;

/* -------------------------------------------------------------------------- */
/*                              struct DynamicMesh                            */
/* -------------------------------------------------------------------------- */
pub struct DynamicMesh {
    vertex_buffer: VertexBuffer,
    index_buffer: IndexBuffer,
    num_indices: u32
}

/* --------------------------------- Getter --------------------------------- */
impl DynamicMesh {
    #[inline]
    pub fn get_vertex_buffer(&self) -> VertexBuffer {
        self.vertex_buffer.clone()
    }

    #[inline]
    pub fn get_index_buffer(&self) -> IndexBuffer {
        self.index_buffer.clone()
    }

    #[inline]
    pub fn get_num_indices(&self) -> u32 {
        self.num_indices
    }
}

/* --------------------------------- Method --------------------------------- */
impl DynamicMesh {
    pub fn new(device: &ID3D12Device, mesh_name: &str) -> DynamicMesh {
        let mut file = std::fs::File::open(mesh_name).unwrap();

        let mut signature: MaybeUninit<[u8; 3]> = MaybeUninit::<[u8; 3]>::uninit();
        let signature = unsafe { signature.assume_init_mut() };
        file.read(signature).unwrap();

        let signature = String::from_utf8(signature.to_vec()).unwrap();
        if signature.as_str().cmp("S3D") != std::cmp::Ordering::Equal {
            panic!("none s3d signature file cannot be read");
        }

        // read vertex data
        let mut num_vertices_buffer: MaybeUninit<[u8; 4]> = MaybeUninit::<[u8; 4]>::uninit();
        let num_vertices_buffer = unsafe { num_vertices_buffer.assume_init_mut() };
        file.read(num_vertices_buffer).unwrap();
        let num_vertices = unsafe { transmute::<[u8; 4], u32>(*num_vertices_buffer) } as usize;

        #[repr(C, packed)]
        struct Vertex {
            position: Float3,
            normal: Float3
        }

        let mut vertices = vec![0; num_vertices * std::mem::size_of::<Vertex>()];
        file.read(&mut vertices).unwrap();
        let vertices = vertices.as_ptr() as *const Vertex;

        // read index data
        let mut num_indices_buffer: MaybeUninit<[u8; 4]> = MaybeUninit::<[u8; 4]>::uninit();
        let num_indices_buffer = unsafe { num_indices_buffer.assume_init_mut() };
        file.read(num_indices_buffer).unwrap();
        let num_indices = unsafe { transmute::<[u8; 4], u32>(*num_indices_buffer) } as usize;

        let mut indices = vec![0; num_indices * std::mem::size_of::<u32>()];
        file.read(&mut indices).unwrap();
        let indices = indices.as_ptr() as *const u32;

        let mut vertex_buffer = VertexBuffer::with_name(
            &device,
            BUFFER_MEMORY_TYPES::BUFFER_MEMORY_TYPE_DYNAMIC,
            num_vertices * std::mem::size_of::<Vertex>(),
            std::mem::size_of::<Vertex>(),
            format!("{}_VertexBuffer", mesh_name).as_str()
        );

        let mut index_buffer = IndexBuffer::with_name(
            &device,
            BUFFER_MEMORY_TYPES::BUFFER_MEMORY_TYPE_DYNAMIC,
            num_indices * std::mem::size_of::<u32>(),
            format!("{}_IndexBuffer", mesh_name).as_str()
        );

        vertex_buffer.copy(vertices, num_vertices);
        index_buffer.copy(indices, num_indices);

        DynamicMesh {
            vertex_buffer,
            index_buffer,
            num_indices: num_indices as u32
        }
    }
}
