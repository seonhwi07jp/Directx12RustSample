#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::{ffi::c_void, rc::Rc};

use windows::{Win32::Graphics::{
    Direct3D12::{
        ID3D12Device, ID3D12Resource, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_UPLOAD, D3D12_MEMORY_POOL_UNKNOWN, D3D12_RESOURCE_DESC,
        D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
    },
    Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC},
}, core::PCWSTR};

/* ---------------------------- enum BUFFER_TYPE ---------------------------- */
#[derive(Clone, Copy)]
pub enum BUFFER_MEMORY_TYPES {
    BUFFER_MEMORY_TYPE_DYNAMIC,
    BUFFER_MEMORY_TYPE_STATIC,
}

/* ---------------------------- enum BUFFER_TYPES --------------------------- */
#[derive(Clone, Copy)]
pub enum BUFFER_DIRECTION_TYPES {
    BUFFER_DIRECTION_TYPE_1D,
    BUFFER_DIRECTION_TYPE_2D,
    BUFFER_DIRECTION_TYPE_3D,
}

/* -------------------------------------------------------------------------- */
/*                                struct Buffer                               */
/* -------------------------------------------------------------------------- */
pub struct Buffer {
    buffer_memory_type: BUFFER_MEMORY_TYPES,
    buffer_direction_type: BUFFER_DIRECTION_TYPES,
    pub(super) resource: ID3D12Resource,
    pub(super) p_map_resource: *const c_void,
    name: Rc<String>,
}

impl Clone for Buffer {
    #[inline]
    fn clone(&self) -> Self {
        Buffer {
            buffer_memory_type: self.buffer_memory_type,
            buffer_direction_type: self.buffer_direction_type,
            resource: self.resource.clone(),
            p_map_resource: self.p_map_resource,
            name: Rc::clone(&self.name),
        }
    }
}

/* --------------------------------- Method --------------------------------- */
impl Buffer {
    pub fn new(
        device: &ID3D12Device, buffer_memory_types: BUFFER_MEMORY_TYPES,
        buffer_direction_types: BUFFER_DIRECTION_TYPES, size: usize,
    ) -> Buffer {
        let heap_properties = match buffer_memory_types {
            BUFFER_MEMORY_TYPES::BUFFER_MEMORY_TYPE_DYNAMIC => D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_UPLOAD,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 0,
                VisibleNodeMask: 0,
            },
            BUFFER_MEMORY_TYPES::BUFFER_MEMORY_TYPE_STATIC => D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 0,
                VisibleNodeMask: 0,
            },
        };

        let resource_desc = match buffer_direction_types {
            BUFFER_DIRECTION_TYPES::BUFFER_DIRECTION_TYPE_1D => D3D12_RESOURCE_DESC {
                Alignment: 0,
                DepthOrArraySize: 1,
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Flags: D3D12_RESOURCE_FLAG_NONE,
                Format: DXGI_FORMAT_UNKNOWN,
                Height: 1,
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Width: size as u64,
            },
            _ => D3D12_RESOURCE_DESC { ..Default::default() },
        };

        let mut resource: Option<ID3D12Resource> = None;
        unsafe {
            device.CreateCommittedResource(
                &heap_properties,
                D3D12_HEAP_FLAG_NONE,
                &resource_desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                std::ptr::null(),
                &mut resource,
            )
        }
        .unwrap();

        let resource = resource.unwrap();
        let mut p_map_resource: *mut c_void = std::ptr::null_mut();

        // if memory type is dynamic. resource is mapped.
        if let BUFFER_MEMORY_TYPES::BUFFER_MEMORY_TYPE_DYNAMIC = buffer_memory_types {
            unsafe { resource.Map(0, std::ptr::null(), &mut p_map_resource) }.unwrap();
        }

        let name = "Default_Buffer\0";
        let name_vec = name.encode_utf16().collect::<Vec<u16>>();
        let name = String::from_utf16(&name_vec).unwrap();
        
        unsafe {resource.SetName(PCWSTR(name_vec.as_ptr()))}.unwrap();

        Buffer {
            buffer_memory_type: buffer_memory_types,
            buffer_direction_type: buffer_direction_types,
            resource,
            p_map_resource,
            name: Rc::new(name)
        }
    }

    pub fn with_name(
        device: &ID3D12Device, buffer_memory_types: BUFFER_MEMORY_TYPES,
        buffer_direction_types: BUFFER_DIRECTION_TYPES, size: usize, name: &str,
    ) -> Buffer {
        let heap_properties = match buffer_memory_types {
            BUFFER_MEMORY_TYPES::BUFFER_MEMORY_TYPE_DYNAMIC => D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_UPLOAD,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 0,
                VisibleNodeMask: 0,
            },
            BUFFER_MEMORY_TYPES::BUFFER_MEMORY_TYPE_STATIC => D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 0,
                VisibleNodeMask: 0,
            },
        };

        let resource_desc = match buffer_direction_types {
            BUFFER_DIRECTION_TYPES::BUFFER_DIRECTION_TYPE_1D => D3D12_RESOURCE_DESC {
                Alignment: 0,
                DepthOrArraySize: 1,
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Flags: D3D12_RESOURCE_FLAG_NONE,
                Format: DXGI_FORMAT_UNKNOWN,
                Height: 1,
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Width: size as u64,
            },
            _ => D3D12_RESOURCE_DESC { ..Default::default() },
        };

        let mut resource: Option<ID3D12Resource> = None;
        unsafe {
            device.CreateCommittedResource(
                &heap_properties,
                D3D12_HEAP_FLAG_NONE,
                &resource_desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                std::ptr::null(),
                &mut resource,
            )
        }
        .unwrap();

        let resource = resource.unwrap();
        let mut p_map_resource: *mut c_void = std::ptr::null_mut();

        // if memory type is dynamic. resource is mapped.
        if let BUFFER_MEMORY_TYPES::BUFFER_MEMORY_TYPE_DYNAMIC = buffer_memory_types {
            unsafe { resource.Map(0, std::ptr::null(), &mut p_map_resource) }.unwrap();
        }

        let mut name_vec = name.encode_utf16().collect::<Vec<u16>>();
        name_vec.push('\0' as u16);
        let name = String::from_utf16(&name_vec).unwrap();

        if cfg!(debug_assertions) {
            unsafe { resource.SetName(PCWSTR(name_vec.as_ptr())) }.unwrap();
        }

        Buffer {
            buffer_memory_type: buffer_memory_types,
            buffer_direction_type: buffer_direction_types,
            resource,
            p_map_resource,
            name: Rc::new(name)
        }
    }

    #[inline]
    pub fn copy<T>(&mut self, p_data: *const T, len: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(p_data, self.p_map_resource as *mut T, len);
        }
    }
}
