#![allow(unused)]

use std::ffi::c_void;

use windows::{
    core::PCWSTR,
    Win32::Graphics::{Direct3D12::*, Dxgi::Common::*},
};

use crate::{core::*, utils::*, shaders::*};

/* ------------------------------ PipelineState ----------------------------- */
pub struct PipelineState {
    pipeline_state: ID3D12PipelineState,
    input_element_descs: Vec<D3D12_INPUT_ELEMENT_DESC>,
}

impl PipelineState {
    #[inline(always)]
    pub(super) fn get_pipeline_state(&self) -> ID3D12PipelineState { self.pipeline_state.clone() }
}

/* ----------------------------- PSOStateBuilder ---------------------------- */
pub struct PSOStateBuilder {
    graphics_pipeline_state_desc: D3D12_GRAPHICS_PIPELINE_STATE_DESC,
}

impl Default for PSOStateBuilder {
    fn default() -> Self {
        PSOStateBuilder {
            graphics_pipeline_state_desc: D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: None,
                VS: Default::default(),
                PS: Default::default(),
                DS: Default::default(),
                HS: Default::default(),
                GS: Default::default(),
                StreamOutput: Default::default(),
                BlendState: D3D12_BLEND_DESC_UTIL { ..Default::default() }.into(),
                SampleMask: D3D12_DEFAULT_SAMPLE_MASK,
                RasterizerState: D3D12_RASTERIZER_DESC_UTIL { ..Default::default() }.into(),
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC_UTIL { ..Default::default() }.into(),
                InputLayout: Default::default(),
                IBStripCutValue: Default::default(),
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: [DXGI_FORMAT_UNKNOWN; 8],
                DSVFormat: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                NodeMask: 0,
                CachedPSO: Default::default(),
                Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
            },
        }
    }
}

/* ---------------------------- PSOShaderBuilder ---------------------------- */
pub struct PSOShaderBuilder {
    graphics_pipeline_state_desc: D3D12_GRAPHICS_PIPELINE_STATE_DESC,
}

/* -------------------------- PipelineStateBuilder -------------------------- */
pub struct PipelineStateBuilder<'a> {
    graphics_pipeline_state_desc: D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    input_element_descs: Vec<D3D12_INPUT_ELEMENT_DESC>,
    name: &'a str,
}

/* -------------------------- impl PSOStateBuilder -------------------------- */
impl PSOStateBuilder {
    pub fn new() -> Self { Default::default() }

    pub fn blend_state(mut self, blend_state: D3D12_BLEND_DESC) -> Self {
        self.graphics_pipeline_state_desc.BlendState = blend_state;

        self
    }

    pub fn rasterizer_state(mut self, rasterlizer_state: D3D12_RASTERIZER_DESC) -> Self {
        self.graphics_pipeline_state_desc.RasterizerState = rasterlizer_state;

        self
    }

    pub fn depth_stencil_state(mut self, depth_stencil_state: D3D12_DEPTH_STENCIL_DESC) -> Self {
        self.graphics_pipeline_state_desc.DepthStencilState = depth_stencil_state;

        self
    }

    pub fn sample_mask(mut self, sample_mask: u32) -> Self {
        self.graphics_pipeline_state_desc.SampleMask = sample_mask;

        self
    }

    pub fn primitive_topology_type(mut self, primitive_topology_type: D3D12_PRIMITIVE_TOPOLOGY_TYPE) -> Self {
        assert!(primitive_topology_type != D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED);

        self.graphics_pipeline_state_desc.PrimitiveTopologyType = primitive_topology_type;

        self
    }

    pub fn rendet_target_format(
        mut self, rtv_format: DXGI_FORMAT, dsv_format: DXGI_FORMAT, msaa_count: u32, msaa_quality: u32,
    ) -> Self {
        self.render_target_formats(vec![rtv_format], dsv_format, msaa_count, msaa_quality)
    }

    pub fn render_target_formats(
        mut self, rtv_formats: Vec<DXGI_FORMAT>, dsv_format: DXGI_FORMAT, msaa_count: u32, msaa_quality: u32,
    ) -> Self {
        assert!(rtv_formats.len() <= 8, "number of rtvs cannot over 8");
        for i in 0..rtv_formats.len() {
            assert!(rtv_formats[i] != DXGI_FORMAT_UNKNOWN);
            self.graphics_pipeline_state_desc.RTVFormats[i] = rtv_formats[i];
        }

        self.graphics_pipeline_state_desc.NumRenderTargets = rtv_formats.len() as u32;
        self.graphics_pipeline_state_desc.DSVFormat = dsv_format;
        self.graphics_pipeline_state_desc.SampleDesc.Count = msaa_count;
        self.graphics_pipeline_state_desc.SampleDesc.Quality = msaa_quality;

        self
    }

    pub fn build_state(self) -> PSOShaderBuilder {
        PSOShaderBuilder {
            graphics_pipeline_state_desc: self.graphics_pipeline_state_desc,
        }
    }
}

/* -------------------------- impl PSOShaderBuilder ------------------------- */
impl PSOShaderBuilder {
    pub fn vertex_shader(mut self, shader: &Shader) -> Self {
        self.graphics_pipeline_state_desc.VS = shader.get_shader_byte_code();

        self
    }

    pub fn pixel_shader(mut self, shader: &Shader) -> Self {
        self.graphics_pipeline_state_desc.PS = shader.get_shader_byte_code();

        self
    }

    pub fn geometry_shader(mut self, shader: &Shader) -> Self {
        self.graphics_pipeline_state_desc.GS = shader.get_shader_byte_code();

        self
    }

    pub fn hull_shader(mut self, shader: &Shader) -> Self {
        self.graphics_pipeline_state_desc.HS = shader.get_shader_byte_code();

        self
    }

    pub fn domain_shader(mut self, shader: &Shader) -> Self {
        self.graphics_pipeline_state_desc.DS = shader.get_shader_byte_code();

        self
    }

    pub fn build_shader(self) -> PipelineStateBuilder<'static> {
        PipelineStateBuilder {
            graphics_pipeline_state_desc: self.graphics_pipeline_state_desc,
            input_element_descs: Default::default(),
            name: "Default_PipelineState",
        }
    }
}

/* ------------------------ impl PipelineStateBuilder ----------------------- */
impl<'a> PipelineStateBuilder<'a> {
    pub fn input_layout(mut self, input_element_descs: Vec<D3D12_INPUT_ELEMENT_DESC>) -> Self {
        self.graphics_pipeline_state_desc.InputLayout.NumElements = input_element_descs.len() as u32;
        self.graphics_pipeline_state_desc.InputLayout.pInputElementDescs = input_element_descs.as_ptr();
        self.input_element_descs = input_element_descs;

        self
    }

    pub fn root_signature(mut self, root_signature: &RootSignature) -> Self {
        self.graphics_pipeline_state_desc.pRootSignature = Some(root_signature.get_root_signature());

        self
    }

    pub fn name(mut self, name: &'a str) -> Self {
        self.name = name;

        self
    }

    pub fn build(self, device: &ID3D12Device) -> PipelineState {
        assert!(
            self.graphics_pipeline_state_desc.DepthStencilState.DepthEnable
                != (self.graphics_pipeline_state_desc.DSVFormat == DXGI_FORMAT_UNKNOWN),
            "dsv format cannot be set with depth enabled"
        );

        let pipeline_state =
            unsafe { device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&self.graphics_pipeline_state_desc) }
                .unwrap();

        if cfg!(debug_assertions) {
            let mut name = self.name.encode_utf16().collect::<Vec<u16>>();
            name.push('\0' as u16);
            unsafe { pipeline_state.SetName(PCWSTR(name.as_ptr())) };
        }

        PipelineState {
            pipeline_state,
            input_element_descs: self.input_element_descs,
        }
    }
}
