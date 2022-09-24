use windows::{Win32::Graphics::Direct3D12::{
    ID3D12DescriptorHeap, ID3D12Device, D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
}, core::PCWSTR};

use crate::buffers::{Buffer, BUFFER_DIRECTION_TYPES, BUFFER_MEMORY_TYPES};

/* -------------------------------------------------------------------------- */
/*                            struct ConstantBuffer                           */
/* -------------------------------------------------------------------------- */
pub struct ConstantBuffer {
    buffer: Buffer,
    gpu_virtual_address: u64,
    descriptor_heap: ID3D12DescriptorHeap,
}

/* --------------------------------- Getter --------------------------------- */
impl ConstantBuffer {
    #[inline]
    pub fn get_descriptor_heap(&self) -> ID3D12DescriptorHeap { self.descriptor_heap.clone() }

    #[inline]
    pub fn get_gpu_virtual_address(&self) -> u64 { self.gpu_virtual_address }
}

/* --------------------------------- Method --------------------------------- */
impl ConstantBuffer {
    pub fn new(device: &ID3D12Device, size: usize) -> ConstantBuffer {
        let buffer = Buffer::with_name(
            &device,
            BUFFER_MEMORY_TYPES::BUFFER_MEMORY_TYPE_DYNAMIC,
            BUFFER_DIRECTION_TYPES::BUFFER_DIRECTION_TYPE_1D,
            size,
            "Default_ConstantBuffer",
        );

        let gpu_virtual_address = unsafe { buffer.resource.GetGPUVirtualAddress() };

        let descriptorh_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
            NumDescriptors: 1,
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        };

        let descriptor_heap: ID3D12DescriptorHeap =
            unsafe { device.CreateDescriptorHeap(&descriptorh_heap_desc) }.unwrap();

        ConstantBuffer {
            buffer,
            gpu_virtual_address,
            descriptor_heap,
        }
    }

    pub fn with_name(device: &ID3D12Device, size: usize, name: &str) -> ConstantBuffer {
        let buffer = Buffer::with_name(
            &device,
            BUFFER_MEMORY_TYPES::BUFFER_MEMORY_TYPE_DYNAMIC,
            BUFFER_DIRECTION_TYPES::BUFFER_DIRECTION_TYPE_1D,
            size,
            name
        );

        let gpu_virtual_address = unsafe { buffer.resource.GetGPUVirtualAddress() };

        let descriptor_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
            NumDescriptors: 1,
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        };

        let descriptor_heap: ID3D12DescriptorHeap =
            unsafe { device.CreateDescriptorHeap(&descriptor_heap_desc) }.unwrap();

        if cfg!(debug_assertions) {
            let name = format!("{}_DescriptorHeap", name);
            let mut name_vec = name.encode_utf16().collect::<Vec<u16>>();
            name_vec.push('\0' as u16);

            unsafe { descriptor_heap.SetName(PCWSTR(name_vec.as_ptr())) }.unwrap();
        }

        ConstantBuffer {
            buffer,
            gpu_virtual_address,
            descriptor_heap,
        }
    }

    #[inline]
    pub fn copy<T>(&mut self, p_data: *const T, len: usize) { self.buffer.copy(p_data, len); }
}
