#![allow(dead_code)]

use std::{
    alloc::{alloc, dealloc, Layout},
    ffi::CString,
};
use windows::{
    core::*,
    Win32::Graphics::{Direct3D::*, Direct3D12::*},
};

pub struct RootParameter {
    parameter: D3D12_ROOT_PARAMETER,
}

impl Default for RootParameter {
    #[inline]
    fn default() -> Self { unsafe { std::mem::zeroed() } }
}

impl Drop for RootParameter {
    #[inline]
    fn drop(&mut self) {
        if self.parameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE {
            unsafe {
                let layout = Layout::array::<D3D12_DESCRIPTOR_RANGE>(
                    self.parameter.Anonymous.DescriptorTable.NumDescriptorRanges as usize,
                )
                .unwrap();

                dealloc(
                    self.parameter.Anonymous.DescriptorTable.pDescriptorRanges as *mut u8,
                    layout,
                );
            }
        }
    }
}

pub struct RootSignature {
    root_signature: ID3D12RootSignature,
}

impl RootSignature {
    #[inline]
    pub fn get_root_signature(&self) -> ID3D12RootSignature { self.root_signature.clone() }
}

// builders
pub struct ConstantsRootParameterBuilder {
    root_parameter: RootParameter,
}

pub struct ConstantBufferRootParameterBuilder {
    root_parameter: RootParameter,
}

pub struct ShaderBufferRootParameterBuilder {
    root_parameter: RootParameter,
}

pub struct DescriptorTableRootParameterBuilder {
    root_parameter: RootParameter,
}

pub struct DescriptorRangeBuilder {
    root_parameter: RootParameter,
}

pub struct RootSignatureBuilder<'a> {
    num_parameters: usize,
    num_static_samplers: usize,
    flags: D3D12_ROOT_SIGNATURE_FLAGS,
    p_parameters: *mut RootParameter,
    p_samplers: *mut D3D12_STATIC_SAMPLER_DESC,
    name: &'a str,
}

impl<'a> Drop for RootSignatureBuilder<'a> {
    #[inline]
    fn drop(&mut self) {
        let parameters_layout = Layout::array::<RootParameter>(self.num_parameters).unwrap();
        let samplers_layout = Layout::array::<D3D12_STATIC_SAMPLER_DESC>(self.num_static_samplers).unwrap();

        unsafe {
            dealloc(self.p_parameters as _, parameters_layout);
            dealloc(self.p_samplers as _, samplers_layout);
        }
    }
}

impl ConstantsRootParameterBuilder {
    #[inline]
    pub fn new() -> Self {
        ConstantsRootParameterBuilder {
            root_parameter: RootParameter {
                parameter: D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                            Num32BitValues: 0,
                        },
                    },
                },
            },
        }
    }

    #[inline]
    pub fn visibility(mut self, visibility: D3D12_SHADER_VISIBILITY) -> Self {
        self.root_parameter.parameter.ShaderVisibility = visibility;

        self
    }

    #[inline]
    pub fn register(mut self, register: u32) -> Self {
        self.root_parameter.parameter.Anonymous.Constants.ShaderRegister = register;

        self
    }

    #[inline]
    pub fn space(mut self, space: u32) -> Self {
        self.root_parameter.parameter.Anonymous.Constants.RegisterSpace = space;

        self
    }

    #[inline]
    pub fn num_32bit_values(mut self, num_32bit_values: u32) -> Self {
        self.root_parameter.parameter.Anonymous.Constants.Num32BitValues = num_32bit_values;

        self
    }

    #[inline]
    pub fn build(self) -> RootParameter { self.root_parameter }
}

impl ConstantBufferRootParameterBuilder {
    #[inline]
    pub fn new() -> Self {
        ConstantBufferRootParameterBuilder {
            root_parameter: RootParameter {
                parameter: D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR {
                            RegisterSpace: 0,
                            ShaderRegister: 0,
                        },
                    },
                },
            },
        }
    }

    #[inline]
    pub fn visibility(mut self, visibility: D3D12_SHADER_VISIBILITY) -> Self {
        self.root_parameter.parameter.ShaderVisibility = visibility;

        self
    }

    #[inline]
    pub fn register(mut self, register: u32) -> Self {
        self.root_parameter.parameter.Anonymous.Descriptor.ShaderRegister = register;

        self
    }

    #[inline]
    pub fn space(mut self, space: u32) -> Self {
        self.root_parameter.parameter.Anonymous.Descriptor.RegisterSpace = space;

        self
    }

    #[inline]
    pub fn build(self) -> RootParameter { self.root_parameter }
}

impl ShaderBufferRootParameterBuilder {
    #[inline]
    pub fn new() -> Self {
        ShaderBufferRootParameterBuilder {
            root_parameter: RootParameter {
                parameter: {
                    D3D12_ROOT_PARAMETER {
                        ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                        Anonymous: D3D12_ROOT_PARAMETER_0 {
                            Descriptor: D3D12_ROOT_DESCRIPTOR {
                                ShaderRegister: 0,
                                RegisterSpace: 0,
                            },
                        },
                    }
                },
            },
        }
    }

    #[inline]
    pub fn visibility(mut self, visibility: D3D12_SHADER_VISIBILITY) -> Self {
        self.root_parameter.parameter.ShaderVisibility = visibility;

        self
    }

    #[inline]
    pub fn register(mut self, register: u32) -> Self {
        self.root_parameter.parameter.Anonymous.Descriptor.ShaderRegister = register;

        self
    }

    #[inline]
    pub fn space(mut self, space: u32) -> Self {
        self.root_parameter.parameter.Anonymous.Descriptor.RegisterSpace = space;

        self
    }

    #[inline]
    pub fn build(self) -> RootParameter { self.root_parameter }
}

impl DescriptorTableRootParameterBuilder {
    #[inline]
    pub fn new(range_count: usize) -> Self {
        let layout = Layout::array::<D3D12_DESCRIPTOR_RANGE>(range_count).unwrap();

        DescriptorTableRootParameterBuilder {
            root_parameter: RootParameter {
                parameter: D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                            NumDescriptorRanges: range_count as u32,
                            pDescriptorRanges: unsafe { alloc(layout).cast() },
                        },
                    },
                },
            },
        }
    }

    #[inline]
    pub fn visibility(mut self, visibility: D3D12_SHADER_VISIBILITY) -> DescriptorTableRootParameterBuilder {
        self.root_parameter.parameter.ShaderVisibility = visibility;

        self
    }

    #[inline]
    pub fn build_table(self) -> DescriptorRangeBuilder {
        DescriptorRangeBuilder {
            root_parameter: self.root_parameter,
        }
    }
}

impl DescriptorRangeBuilder {
    #[inline]
    pub fn descriptor_range(
        self, range_index: u32, range_type: D3D12_DESCRIPTOR_RANGE_TYPE, register: u32, count: u32, space: u32,
    ) -> Self {
        unsafe {
            let descriptor_table = &self.root_parameter.parameter.Anonymous.DescriptorTable;
            if descriptor_table.NumDescriptorRanges <= range_index {
                panic!("range_index is out_of_range!");
            }

            let range = descriptor_table.pDescriptorRanges.offset(range_index as isize) as *mut D3D12_DESCRIPTOR_RANGE;
            let mut range = range.as_mut().unwrap();

            range.RangeType = range_type;
            range.BaseShaderRegister = register;
            range.NumDescriptors = count;
            range.RegisterSpace = space;
            range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        }

        self
    }

    #[inline]
    pub fn build(self) -> RootParameter { self.root_parameter }
}

impl<'a> RootSignatureBuilder<'a> {
    #[inline]
    pub fn new(num_parameters: usize, num_static_samplers: usize) -> Self {
        let parameters_layout = Layout::array::<RootParameter>(num_parameters).unwrap();
        let samplers_layout = Layout::array::<D3D12_STATIC_SAMPLER_DESC>(num_static_samplers).unwrap();

        let p_parameters = unsafe { alloc(parameters_layout) } as *mut RootParameter;
        let p_samplers = unsafe { alloc(samplers_layout) } as *mut D3D12_STATIC_SAMPLER_DESC;

        RootSignatureBuilder {
            num_parameters,
            num_static_samplers,
            p_parameters,
            p_samplers,
            flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
            name: "Default_RootSignature",
        }
    }

    #[inline(always)]
    pub fn parameter(self, parameter_index: usize, parameter: RootParameter) -> Self {
        if parameter_index >= self.num_parameters {
            panic!("parameter_index is out_of_range");
        }

        unsafe {
            let p_parameter = self.p_parameters.offset(parameter_index as isize);
            p_parameter.write(parameter);
        }

        self
    }

    #[inline(always)]
    pub fn flags(mut self, flags: D3D12_ROOT_SIGNATURE_FLAGS) -> Self {
        self.flags = flags;

        self
    }

    #[inline(always)]
    pub fn name(mut self, name: &'a str) -> Self {
        self.name = name;

        self
    }

    pub fn build(self, device: &ID3D12Device) -> RootSignature {
        let root_signature_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: self.num_parameters as u32,
            pParameters: self.p_parameters as *const D3D12_ROOT_PARAMETER,
            NumStaticSamplers: self.num_static_samplers as u32,
            pStaticSamplers: self.p_samplers,
            Flags: self.flags,
        };

        let mut result_blob: Option<ID3DBlob> = None;
        let mut error_blob: Option<ID3DBlob> = None;
        if unsafe {
            D3D12SerializeRootSignature(
                &root_signature_desc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut result_blob,
                &mut error_blob,
            )
        }
        .is_err()
        {
            let p_error_message = unsafe { error_blob.unwrap().GetBufferPointer() } as *mut i8;
            let error_message = unsafe { CString::from_raw(p_error_message) };
            let error_message = error_message.to_string_lossy();

            panic!("error_message : {}", error_message);
        }

        let result_blob = result_blob.unwrap();
        let root_signature = unsafe {
            device.CreateRootSignature::<ID3D12RootSignature>(
                0,
                std::slice::from_raw_parts(result_blob.GetBufferPointer() as *const u8, result_blob.GetBufferSize()),
            )
        }
        .unwrap();

        if cfg!(debug_assertions) {
            let mut name = self.name.encode_utf16().collect::<Vec<u16>>();
            name.push('\0' as u16);
            unsafe { root_signature.SetName(PCWSTR(name.as_ptr())) }.unwrap();
        }

        RootSignature { root_signature }
    }
}
