use std::borrow::Cow;
use std::ffi::{OsString, OsStr};
use std::os::windows::ffi::EncodeWide;
use std::os::windows::prelude::{OsStringExt, OsStrExt};
use std::str::FromStr;

use ssun_math::Matrix4x4;
use ssun_renderer::{core::*, buffers::*, shaders::*, utils::*, meshs::*, input_layout};
use utf16_literal::utf16;

use windows::core::{Result, PCWSTR};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::Graphics::Direct3D::Fxc::{D3DCOMPILE_DEBUG, D3DCOMPILE_SKIP_OPTIMIZATION};
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, D3D12_INPUT_ELEMENT_DESC,
    D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DispatchMessageW, LoadCursorW, PeekMessageW, PostQuitMessage, RegisterClassExW,
    ShowWindow, TranslateMessage, CS_HREDRAW, CS_VREDRAW, CW_USEDEFAULT, IDC_ARROW, MSG, PM_REMOVE, SW_SHOW,
    WM_DESTROY, WM_QUIT, WNDCLASSEXW, WS_OVERLAPPEDWINDOW,
};

static WINDOW_WIDTH: u32 = 1280;
static WINDOW_HEIGHT: u32 = 720;

extern "system" fn wnd_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    match msg {
        WM_DESTROY => {
            unsafe { PostQuitMessage(0) };
            LRESULT(0)
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

fn main() -> Result<()> {
    let pix = WindowPIX::new();
    pix.enable();
    
    let instance = unsafe { GetModuleHandleW(None) }.unwrap();
    debug_assert!(instance.0 != 0);

    let wc = WNDCLASSEXW {
        cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
        style: CS_HREDRAW | CS_VREDRAW,
        lpfnWndProc: Some(wnd_proc),
        hInstance: instance,
        hCursor: unsafe { LoadCursorW(None, IDC_ARROW)? },
        lpszClassName: PCWSTR(utf16!("RustWindowClass\0").as_ptr()),
        ..Default::default()
    };

    unsafe { RegisterClassExW(&wc) };

    let hwnd = unsafe {
        CreateWindowExW(
            Default::default(),
            PCWSTR(utf16!("RustWindowClass\0").as_ptr()),
            PCWSTR(utf16!("RustWindowClass\0").as_ptr()),
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            WINDOW_WIDTH as i32,
            WINDOW_HEIGHT as i32,
            None,
            None,
            instance,
            std::ptr::null_mut(),
        )
    };
    debug_assert!(hwnd.0 != 0);

    let (mut renderer, rendering_context) = Renderer::new(hwnd, WINDOW_WIDTH, WINDOW_HEIGHT);
    let device = renderer.get_device();

    let constant_buffer = ConstantBufferRootParameterBuilder::new().register(0).space(0).build();

    let root_signature = RootSignatureBuilder::new(1, 0)
        .flags(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)
        .parameter(0, constant_buffer)
        .build(&device);

    let vs_shader = ShaderBuilder::new("../../shaders/shader.hlsl")
        .entry_point(b"VSMain\0")
        .target(b"vs_5_0\0")
        .flags(D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION)
        .build();

    let ps_shader = ShaderBuilder::new("../../shaders/shader.hlsl")
        .entry_point(b"PSMain\0")
        .target(b"ps_5_0\0")
        .flags(D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION)
        .build();

    let pso = PSOStateBuilder::new()
        .rasterizer_state(D3D12_RASTERIZER_DESC_UTIL { ..Default::default() }.into())
        .blend_state(D3D12_BLEND_DESC_UTIL { ..Default::default() }.into())
        .depth_stencil_state(
            D3D12_DEPTH_STENCIL_DESC_UTIL {
                DepthEnable: TRUE,
                ..Default::default()
            }
                .into(),
        )
        .rendet_target_format(DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_D32_FLOAT, 1, 0)
        .primitive_topology_type(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE)
        .build_state()
        .vertex_shader(&vs_shader)
        .pixel_shader(&ps_shader)
        .build_shader()
        .root_signature(&root_signature)
        .input_layout(vec![
            input_layout!(
                b"POSITION\0",
                0,
                DXGI_FORMAT_R32G32B32_FLOAT,
                0,
                D3D12_APPEND_ALIGNED_ELEMENT,
                D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                0
            ),
            input_layout!(
                b"NORMAL\0",
                0,
                DXGI_FORMAT_R32G32B32A32_FLOAT,
                0,
                D3D12_APPEND_ALIGNED_ELEMENT,
                D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                0
            ),
        ])
        .build(&device);

    let magician_mesh = DynamicMesh::new(&device, "../../magician.s3d");

    let projection = Matrix4x4::perspective_by_deg(90.0, WINDOW_WIDTH as f32 / WINDOW_HEIGHT as f32, 0.1, 100.0);

    let mut constant_buffer = ConstantBuffer::with_name(&device, std::mem::size_of::<Matrix4x4>(), "Matrix_ConstantBuffer");

    unsafe { ShowWindow(hwnd, SW_SHOW) };

    let mut deg: f32 = 0.0;
    let mut msg = MSG::default();
    loop {
        if unsafe { PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE) }.into() {
            unsafe {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            if msg.message == WM_QUIT {
                break;
            }
        }

        deg += 1.0;
        let world =
            Matrix4x4::scale(0.3, 0.3, 0.3) * Matrix4x4::rotate_y_by_deg(deg) * Matrix4x4::translate(0.0, 0.0, 1.0);
        let wvp = world * projection;
        constant_buffer.copy(&wvp, 1);

        renderer.begin_render().unwrap();

        rendering_context.set_root_signature(&root_signature);
        rendering_context.set_pipeline_state(&pso);
        rendering_context.set_primitive_topology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        rendering_context.set_render_target_with_depth_stencil_buffer(
            renderer.get_final_render_target(),
            renderer.get_final_depth_stencil_buffer(),
        );
        rendering_context.set_constant_buffer(0, &constant_buffer);
        rendering_context.draw_mesh(&magician_mesh);

        renderer.end_render().unwrap();
    }

    Ok(())
}
