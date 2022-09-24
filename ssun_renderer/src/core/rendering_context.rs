#![allow(dead_code)]

use windows::Win32::Graphics::{Direct3D::*, Direct3D12::*};

use crate::{buffers::*, core::*, utils::*, meshs::*};

/* ------------------------- struct RenderingContext ------------------------ */
pub struct RenderingContext {
    command_list: ID3D12GraphicsCommandList,
}

impl RenderingContext {
    #[inline]
    pub fn new(command_list: ID3D12GraphicsCommandList) -> RenderingContext { RenderingContext { command_list } }

    #[inline]
    pub fn set_root_signature(&self, root_signature: &RootSignature) {
        unsafe { self.command_list.SetGraphicsRootSignature(root_signature.get_root_signature()) };
    }

    #[inline]
    pub fn set_pipeline_state(&self, pipeline_state: &PipelineState) {
        unsafe { self.command_list.SetPipelineState(pipeline_state.get_pipeline_state()) };
    }

    #[inline]
    pub fn set_vertex_buffer(&self, vertex_buffer: &VertexBuffer) {
        unsafe { self.command_list.IASetVertexBuffers(0, &[vertex_buffer.get_vertex_buffer_view()]) };
    }

    #[inline]
    pub fn set_index_buffer(&self, index_buffer: &IndexBuffer) {
        unsafe {
            self.command_list.IASetIndexBuffer(&index_buffer.get_index_buffer_view());
        }
    }

    #[inline]
    pub fn set_render_target(&self, rtv_cpu_descriptor_handle: D3D12_CPU_DESCRIPTOR_HANDLE) {
        unsafe { self.command_list.OMSetRenderTargets(1, &rtv_cpu_descriptor_handle, FALSE, std::ptr::null()) };
    }

    #[inline]
    pub fn set_render_target_with_depth_stencil_buffer(
        &self, rtv_cpu_descriptor_handle: D3D12_CPU_DESCRIPTOR_HANDLE,
        dsv_cpu_descriptor_handle: D3D12_CPU_DESCRIPTOR_HANDLE,
    ) {
        unsafe {
            self.command_list
                .OMSetRenderTargets(1, &rtv_cpu_descriptor_handle, FALSE, &dsv_cpu_descriptor_handle);
        }
    }

    #[inline]
    pub fn set_primitive_topology(&self, primitve_topology_type: D3D_PRIMITIVE_TOPOLOGY) {
        unsafe { self.command_list.IASetPrimitiveTopology(primitve_topology_type) };
    }

    #[inline]
    pub fn set_constant_buffer(&self, parameter_index: u32, constant_buffer: &ConstantBuffer) {
        unsafe {
            self.command_list.SetDescriptorHeaps(&[Some(constant_buffer.get_descriptor_heap())]);
            self.command_list
                .SetGraphicsRootConstantBufferView(parameter_index, constant_buffer.get_gpu_virtual_address());
        }
    }

    #[inline]
    pub fn draw(&self) {
        unsafe {
            self.command_list.DrawInstanced(3, 1, 0, 0);
        }
    }

    #[inline]
    pub fn draw_mesh(&self, mesh: &DynamicMesh) {
        let vbv = mesh.get_vertex_buffer().get_vertex_buffer_view();
        let ibv = mesh.get_index_buffer().get_index_buffer_view();

        unsafe {
            self.command_list.IASetVertexBuffers(0, &[vbv]);
            self.command_list.IASetIndexBuffer(&ibv);
            self.command_list.DrawIndexedInstanced(mesh.get_num_indices(), 1, 0, 0, 0);
        }
    }
}
