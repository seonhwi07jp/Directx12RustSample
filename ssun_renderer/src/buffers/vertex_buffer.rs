#![allow(dead_code)]

use windows::Win32::Graphics::Direct3D12::{ID3D12Device, D3D12_VERTEX_BUFFER_VIEW};

use crate::buffers::{Buffer, BUFFER_DIRECTION_TYPES, BUFFER_MEMORY_TYPES};

/* --------------------------- struct VertexBuffer -------------------------- */
pub struct VertexBuffer {
    buffer: Buffer,
    vertex_buffer_view: D3D12_VERTEX_BUFFER_VIEW,
}

impl Clone for VertexBuffer {
    #[inline]
    fn clone(&self) -> Self {
        VertexBuffer {
            buffer: self.buffer.clone(),
            vertex_buffer_view: self.vertex_buffer_view.clone(),
        }
    }
}

/* --------------------------------- Getter --------------------------------- */
impl VertexBuffer {
    #[inline]
    pub(crate) fn get_vertex_buffer_view(&self) -> D3D12_VERTEX_BUFFER_VIEW { self.vertex_buffer_view.clone() }
}

/* --------------------------------- Method --------------------------------- */
impl VertexBuffer {
    pub fn new(
        device: &ID3D12Device, buffer_memory_type: BUFFER_MEMORY_TYPES, size: usize, stride: usize,
    ) -> VertexBuffer {
        let buffer = Buffer::with_name(
            &device,
            buffer_memory_type,
            BUFFER_DIRECTION_TYPES::BUFFER_DIRECTION_TYPE_1D,
            size,
            "Default_VertexBuffer"
        );
        let vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: unsafe { buffer.resource.GetGPUVirtualAddress() },
            SizeInBytes: size as u32,
            StrideInBytes: stride as u32,
        };

        VertexBuffer {
            buffer,
            vertex_buffer_view,
        }
    }

    pub fn with_name(
        device: &ID3D12Device, buffer_memory_type: BUFFER_MEMORY_TYPES, size: usize, stride: usize, name: &str,
    ) -> VertexBuffer {
        let buffer = Buffer::with_name(
            &device,
            buffer_memory_type,
            BUFFER_DIRECTION_TYPES::BUFFER_DIRECTION_TYPE_1D,
            size,
            name,
        );
        let vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: unsafe { buffer.resource.GetGPUVirtualAddress() },
            SizeInBytes: size as u32,
            StrideInBytes: stride as u32,
        };

        VertexBuffer {
            buffer,
            vertex_buffer_view,
        }
    }

    #[inline]
    pub fn copy<T>(&mut self, p_data: *const T, len: usize) { self.buffer.copy(p_data, len); }
}
