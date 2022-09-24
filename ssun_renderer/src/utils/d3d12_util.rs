#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use windows::Win32::{
    Foundation::BOOL, Graphics::Direct3D12::{D3D12_FILL_MODE, D3D12_CULL_MODE, D3D12_CONSERVATIVE_RASTERIZATION_MODE, D3D12_FILL_MODE_SOLID, D3D12_CULL_MODE_BACK, D3D12_DEFAULT_DEPTH_BIAS, D3D12_DEFAULT_DEPTH_BIAS_CLAMP, D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS, D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF, D3D12_RASTERIZER_DESC, D3D12_RENDER_TARGET_BLEND_DESC, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP, D3D12_COLOR_WRITE_ENABLE_ALL, D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT, D3D12_BLEND_DESC, D3D12_DEPTH_WRITE_MASK, D3D12_COMPARISON_FUNC, D3D12_DEPTH_STENCILOP_DESC, D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS, D3D12_DEPTH_WRITE_MASK_ALL, D3D12_COMPARISON_FUNC_LESS, D3D12_DEFAULT_STENCIL_READ_MASK, D3D12_DEFAULT_STENCIL_WRITE_MASK, D3D12_DEPTH_STENCIL_DESC},
};

pub const TRUE: BOOL = BOOL(1);
pub const FALSE: BOOL = BOOL(0);

#[macro_export]
macro_rules! input_layout {
    ($semantic_name:expr, $semantic_index:expr, $format:expr, $input_slot:expr, $aligned_byte_offset:expr, $input_slot_class:expr, $instance_data_step_rate:expr) => {
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: ::windows::core::PCSTR($semantic_name.as_ptr()),
            SemanticIndex: $semantic_index,
            Format: $format,
            InputSlot: $input_slot,
            AlignedByteOffset: $aligned_byte_offset,
            InputSlotClass: $input_slot_class,
            InstanceDataStepRate: $instance_data_step_rate
        }
    };
}

/* ---------------------- struct D3D12_RASTERIZER_DESC_UTILITY --------------------- */
pub struct D3D12_RASTERIZER_DESC_UTIL {
    pub FillMode: D3D12_FILL_MODE,
    pub CullMode: D3D12_CULL_MODE,
    pub FrontCounterClockwise: BOOL,
    pub DepthBias: i32,
    pub DepthBiasClamp: f32,
    pub SlopeScaledDepthBias: f32,
    pub DepthClipEnable: BOOL,
    pub MultisampleEnable: BOOL,
    pub AntialiasedLineEnable: BOOL,
    pub ForcedSampleCount: u32,
    pub ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE,
}

impl Default for D3D12_RASTERIZER_DESC_UTIL {
    #[inline(always)]
    fn default() -> Self {
        D3D12_RASTERIZER_DESC_UTIL {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_BACK,
            FrontCounterClockwise: FALSE,
            DepthBias: D3D12_DEFAULT_DEPTH_BIAS,
            DepthBiasClamp: D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
            SlopeScaledDepthBias: D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS,
            DepthClipEnable: TRUE,
            MultisampleEnable: FALSE,
            AntialiasedLineEnable: FALSE,
            ForcedSampleCount: 0,
            ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
        }
    }
}

impl From<D3D12_RASTERIZER_DESC> for D3D12_RASTERIZER_DESC_UTIL {
    #[inline(always)]
    fn from(item: D3D12_RASTERIZER_DESC) -> Self {
        unsafe { std::mem::transmute::<D3D12_RASTERIZER_DESC, D3D12_RASTERIZER_DESC_UTIL>(item) }
    }
}

impl Into<D3D12_RASTERIZER_DESC> for D3D12_RASTERIZER_DESC_UTIL {
    #[inline(always)]
    fn into(self) -> D3D12_RASTERIZER_DESC {
        unsafe { std::mem::transmute::<D3D12_RASTERIZER_DESC_UTIL, D3D12_RASTERIZER_DESC>(self) }
    }
}

/* -------------------------- struct D3D12_BLEND_DESC_UTIL ------------------------- */
pub struct D3D12_BLEND_DESC_UTIL {
    pub AlphaToCoverageEnable: BOOL,
    pub IndependentBlendEnable: BOOL,
    pub RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC; 8],
}

impl Default for D3D12_BLEND_DESC_UTIL {
    fn default() -> Self {
        let mut item = D3D12_BLEND_DESC_UTIL {
            AlphaToCoverageEnable: FALSE,
            IndependentBlendEnable: FALSE,
            RenderTarget: Default::default(),
        };

        const DEFAULT_RENDER_TARGET_BLEND_DESC: D3D12_RENDER_TARGET_BLEND_DESC = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: FALSE,
            LogicOpEnable: FALSE,
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
        };

        for i in 0..D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT as usize {
            item.RenderTarget[i] = DEFAULT_RENDER_TARGET_BLEND_DESC;
        }

        item
    }
}

impl From<D3D12_BLEND_DESC> for D3D12_BLEND_DESC_UTIL {
    fn from(item: D3D12_BLEND_DESC) -> Self {
        unsafe { std::mem::transmute::<D3D12_BLEND_DESC, D3D12_BLEND_DESC_UTIL>(item) }
    }
}

impl Into<D3D12_BLEND_DESC> for D3D12_BLEND_DESC_UTIL {
    fn into(self) -> D3D12_BLEND_DESC {
        unsafe { std::mem::transmute::<D3D12_BLEND_DESC_UTIL, D3D12_BLEND_DESC>(self) }
    }
}

/* ------------------------ struct D3D12_DEPTH_STENCIL_DESC_UTIL ------------------------ */
pub struct D3D12_DEPTH_STENCIL_DESC_UTIL {
    pub DepthEnable: BOOL,
    pub DepthWriteMask: D3D12_DEPTH_WRITE_MASK,
    pub DepthFunc: D3D12_COMPARISON_FUNC,
    pub StencilEnable: BOOL,
    pub StencilReadMask: u8,
    pub StencilWriteMask: u8,
    pub FrontFace: D3D12_DEPTH_STENCILOP_DESC,
    pub BackFace: D3D12_DEPTH_STENCILOP_DESC,
}

impl Default for D3D12_DEPTH_STENCIL_DESC_UTIL {
    fn default() -> Self {
        const DEFAULT_STENCIL_OP: D3D12_DEPTH_STENCILOP_DESC = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };

        D3D12_DEPTH_STENCIL_DESC_UTIL {
            DepthEnable: TRUE,
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D12_COMPARISON_FUNC_LESS,
            StencilEnable: FALSE,
            StencilReadMask: D3D12_DEFAULT_STENCIL_READ_MASK as u8,
            StencilWriteMask: D3D12_DEFAULT_STENCIL_WRITE_MASK as u8,
            FrontFace: DEFAULT_STENCIL_OP,
            BackFace: DEFAULT_STENCIL_OP,
        }
    }
}

impl From<D3D12_DEPTH_STENCIL_DESC> for D3D12_DEPTH_STENCIL_DESC_UTIL {
    fn from(item: D3D12_DEPTH_STENCIL_DESC) -> Self {
        unsafe { std::mem::transmute::<D3D12_DEPTH_STENCIL_DESC, D3D12_DEPTH_STENCIL_DESC_UTIL>(item) }
    }
}

impl Into<D3D12_DEPTH_STENCIL_DESC> for D3D12_DEPTH_STENCIL_DESC_UTIL {
    fn into(self) -> D3D12_DEPTH_STENCIL_DESC {
        unsafe { std::mem::transmute::<D3D12_DEPTH_STENCIL_DESC_UTIL, D3D12_DEPTH_STENCIL_DESC>(self) }
    }
}
