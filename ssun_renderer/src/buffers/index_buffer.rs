#![allow(dead_code)]

use windows::Win32::Graphics::{
    Direct3D12::{ID3D12Device, D3D12_INDEX_BUFFER_VIEW},
    Dxgi::Common::DXGI_FORMAT_R32_UINT,
};

use crate::buffers::{Buffer, BUFFER_DIRECTION_TYPES, BUFFER_MEMORY_TYPES};

/* -------------------------------------------------------------------------- */
/*                             struct IndexBuffer                             */
/* -------------------------------------------------------------------------- */
pub struct IndexBuffer {
    buffer: Buffer,
    index_buffer_view: D3D12_INDEX_BUFFER_VIEW,
}

impl Clone for IndexBuffer {
    #[inline]
    fn clone(&self) -> Self {
        IndexBuffer {
            buffer: self.buffer.clone(),
            index_buffer_view: self.index_buffer_view.clone(),
        }
    }
}

/* --------------------------------- Getter --------------------------------- */
impl IndexBuffer {
    #[inline]
    pub(crate) fn get_index_buffer_view(&self) -> D3D12_INDEX_BUFFER_VIEW { self.index_buffer_view.clone() }
}

/* --------------------------------- Method --------------------------------- */
impl IndexBuffer {
    pub fn new(device: &ID3D12Device, buffer_memory_type: BUFFER_MEMORY_TYPES, size: usize) -> IndexBuffer {
        let buffer = Buffer::with_name(
            &device,
            buffer_memory_type,
            BUFFER_DIRECTION_TYPES::BUFFER_DIRECTION_TYPE_1D,
            size,
            "Default_IndexBuffer"
        );

        let index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: unsafe { buffer.resource.GetGPUVirtualAddress() },
            SizeInBytes: size as u32,
            Format: DXGI_FORMAT_R32_UINT,
        };

        IndexBuffer {
            buffer,
            index_buffer_view,
        }
    }
    
    pub fn with_name(device: &ID3D12Device, buffer_memory_type: BUFFER_MEMORY_TYPES, size: usize, name: &str) -> IndexBuffer {
        let buffer = Buffer::with_name(
            &device,
            buffer_memory_type,
            BUFFER_DIRECTION_TYPES::BUFFER_DIRECTION_TYPE_1D,
            size,
            name
        );

        let index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: unsafe { buffer.resource.GetGPUVirtualAddress() },
            SizeInBytes: size as u32,
            Format: DXGI_FORMAT_R32_UINT,
        };

        IndexBuffer {
            buffer,
            index_buffer_view,
        }
    }

    #[inline]
    pub fn copy<T>(&mut self, p_data: *const T, len: usize) { self.buffer.copy(p_data, len); }
}
