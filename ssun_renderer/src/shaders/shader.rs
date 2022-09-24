#![allow(dead_code)]
use std::ffi::CString;

use windows::{
    core::{PCSTR, PCWSTR},
    Win32::Graphics::{
        Direct3D::{Fxc::D3DCompileFromFile, ID3DBlob, D3D_SHADER_MACRO},
        Direct3D12::D3D12_SHADER_BYTECODE,
    },
};

/* ------------------------------ struct Shader ----------------------------- */
pub struct Shader {
    shader_byte_code: D3D12_SHADER_BYTECODE,
    shader_blob: Option<ID3DBlob>,
}

impl Default for Shader {
    fn default() -> Self {
        Shader {
            shader_byte_code: Default::default(),
            shader_blob: None,
        }
    }
}

impl Shader {
    #[inline(always)]
    pub(in crate) fn get_shader_byte_code(&self) -> D3D12_SHADER_BYTECODE {
        self.shader_byte_code.clone()
    }
}

/* -------------------------- struct ShaderBuilder -------------------------- */
pub struct ShaderBuilder<'a> {
    file_name: &'a str,
    p_defines: *const D3D_SHADER_MACRO,
    entry_point: &'a [u8],
    target: &'a [u8],
    flags: u32,
}

impl Default for ShaderBuilder<'static> {
    fn default() -> Self {
        ShaderBuilder {
            file_name: " ",
            p_defines: std::ptr::null(),
            entry_point: &['\0' as u8],
            target: &['\0' as u8],
            flags: 0,
        }
    }
}

impl<'a> ShaderBuilder<'a> {
    pub fn new(file_name: &'a str) -> Self {
        ShaderBuilder {
            file_name,
            ..Default::default()
        }
    }

    pub fn entry_point(mut self, entry_point: &'a [u8]) -> Self {
        self.entry_point = entry_point;

        self
    }

    pub fn target(mut self, target: &'a [u8]) -> Self {
        self.target = target;

        self
    }

    pub fn flags(mut self, flags: u32) -> Self {
        self.flags = flags;

        self
    }

    pub fn build(self) -> Shader {
        let mut file_name = self.file_name.encode_utf16().collect::<Vec<u16>>();
        file_name.push('\0' as u16);

        let mut result_blob: Option<ID3DBlob> = None;
        let mut error_blob: Option<ID3DBlob> = None;

        if unsafe {
            D3DCompileFromFile(
                PCWSTR(file_name.as_ptr()),
                self.p_defines,
                None,
                PCSTR(self.entry_point.as_ptr()),
                PCSTR(self.target.as_ptr()),
                self.flags,
                0,
                &mut result_blob,
                &mut error_blob,
            )
        }
            .is_err()
        {
            let p_error_message = unsafe { error_blob.expect("file not found").GetBufferPointer() } as *mut i8;
            let error_message = unsafe { CString::from_raw(p_error_message) };
            let error_message = error_message.to_string_lossy();

            panic!("error_message : {}", error_message);
        }

        let result_blob = result_blob.unwrap();

        Shader {
            shader_byte_code: D3D12_SHADER_BYTECODE {
                BytecodeLength: unsafe { result_blob.GetBufferSize() },
                pShaderBytecode: unsafe { result_blob.GetBufferPointer() },
            },
            shader_blob: Some(result_blob),
        }
    }
}
