use utf16_literal::utf16;
use windows::{
    core::{Interface, Result, PCWSTR},
    Win32::{
        Foundation::{HANDLE, HWND, RECT},
        Graphics::{
            Direct3D::D3D_FEATURE_LEVEL_12_1,
            Direct3D12::{
                D3D12CreateDevice, D3D12GetDebugInterface, ID3D12CommandAllocator, ID3D12CommandList,
                ID3D12CommandQueue, ID3D12Debug, ID3D12DescriptorHeap, ID3D12Device, ID3D12Fence,
                ID3D12GraphicsCommandList, ID3D12Resource, D3D12_CLEAR_FLAG_DEPTH, D3D12_CLEAR_VALUE,
                D3D12_CLEAR_VALUE_0, D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC,
                D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_COMMAND_QUEUE_PRIORITY_NORMAL, D3D12_CPU_DESCRIPTOR_HANDLE,
                D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_DEPTH_STENCIL_VALUE, D3D12_DESCRIPTOR_HEAP_DESC,
                D3D12_DESCRIPTOR_HEAP_FLAG_NONE, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                D3D12_FENCE_FLAG_NONE, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT,
                D3D12_MAX_DEPTH, D3D12_MEMORY_POOL_UNKNOWN, D3D12_MIN_DEPTH, D3D12_RESOURCE_BARRIER,
                D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAG_NONE,
                D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_TRANSITION_BARRIER,
                D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_VIEWPORT,
            },
            Dxgi::{
                Common::{self, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC},
                CreateDXGIFactory2, IDXGIFactory4, IDXGISwapChain3, DXGI_CREATE_FACTORY_DEBUG, DXGI_SWAP_CHAIN_DESC1,
                DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT,
            },
        },
        System::{
            Threading::{CreateEventA, WaitForSingleObject},
            WindowsProgramming::INFINITE,
        },
    },
};

use crate::core::*;

const BACK_BUFFER_COUNT: usize = 2;

#[allow(unused)]
pub struct Renderer {
    hwnd: HWND,

    frame_count: usize,
    window_width: u32,
    window_height: u32,

    current_back_buffer_index: usize,

    dxgi_factory: IDXGIFactory4,
    device: ID3D12Device,
    swap_chain: IDXGISwapChain3,
    fence: ID3D12Fence,
    fence_value: u64,
    fence_event: HANDLE,

    render_targets: [ID3D12Resource; BACK_BUFFER_COUNT],
    rtv_descriptor_heap: ID3D12DescriptorHeap,
    rtv_descriptor_size: usize,

    depth_stencil_buffer: ID3D12Resource,
    dsv_descriptor_heap: ID3D12DescriptorHeap,
    dsv_descriptor_size: usize,

    view_port: D3D12_VIEWPORT,
    scissor_rect: RECT,

    command_allocator: ID3D12CommandAllocator,
    command_queue: ID3D12CommandQueue,
    command_list: ID3D12GraphicsCommandList,
}

impl Renderer {
    #[inline]
    pub fn get_device(&self) -> ID3D12Device { self.device.clone() }

    #[inline]
    pub fn get_final_render_target(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let mut cpu_descriptor_handle = unsafe { self.rtv_descriptor_heap.GetCPUDescriptorHandleForHeapStart() };
        cpu_descriptor_handle.ptr += self.current_back_buffer_index * self.rtv_descriptor_size;

        cpu_descriptor_handle
    }

    #[inline]
    pub fn get_final_depth_stencil_buffer(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        unsafe { self.dsv_descriptor_heap.GetCPUDescriptorHandleForHeapStart() }
    }
}

impl Renderer {
    pub fn new(hwnd: HWND, window_width: u32, window_height: u32) -> (Renderer, RenderingContext) {
        // enable debug layer
        {
            let mut debug_interface: Option<ID3D12Debug> = None;
            unsafe {
                if let Some(debug_interface) = D3D12GetDebugInterface(&mut debug_interface).ok().and(debug_interface) {
                    debug_interface.EnableDebugLayer();
                }
            }
        }

        // create dxgi factory
        let dxgi_factory = unsafe { CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_DEBUG) }.unwrap();

        // select adapter and create device
        let mut idx: u32 = 0;
        #[allow(unused_assignments)]
        let device = loop {
            let adapter = unsafe { dxgi_factory.EnumAdapters1(idx) }.unwrap();
            idx += 1;

            let mut device: Option<ID3D12Device> = None;
            if unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_1, &mut device) }.is_ok() {
                break device.unwrap();
            } else {
                panic!();
            }
        };

        // create command resource
        let (command_allocator, command_queue, command_list) = Renderer::create_command_resource(&device).unwrap();

        // create swap chain
        let swap_chain =
            Renderer::create_swap_chain(hwnd, &dxgi_factory, &command_queue, window_width, window_height).unwrap();
        let current_back_buffer_index = unsafe { swap_chain.GetCurrentBackBufferIndex() } as usize;

        // create rtv_descriptor_heap
        let rtv_descriptor_heap = Renderer::create_rtv_descriptor_heap(&device).unwrap();

        // create render_target_view
        let (render_targets, rtv_descriptor_size) =
            Renderer::create_render_target_views(&device, &swap_chain, unsafe {
                rtv_descriptor_heap.GetCPUDescriptorHandleForHeapStart()
            })
            .unwrap();

        // create dsv_descriptro_heap
        let dsv_descriptor_heap = Renderer::create_dsv_descriptor_heap(&device).unwrap();

        // create depth_stencil_buffer_and_view
        let (depth_stencil_buffer, dsv_descriptor_size) =
            Renderer::create_depth_stencil_buffer_and_view(&device, window_width, window_height, unsafe {
                dsv_descriptor_heap.GetCPUDescriptorHandleForHeapStart()
            })
            .unwrap();

        // create fence
        let (fence_value, fence, fence_event) = Renderer::create_fence(&device).unwrap();

        // create viewport and scissors_rect
        let (view_port, scissor_rect) = Renderer::create_viewport_and_scissor_rect(window_width, window_height);

        (
            Renderer {
                hwnd,

                frame_count: BACK_BUFFER_COUNT,
                window_width,
                window_height,

                current_back_buffer_index,

                dxgi_factory,
                device,
                swap_chain,

                fence_value,
                fence,
                fence_event,

                render_targets,
                rtv_descriptor_heap,
                rtv_descriptor_size,

                dsv_descriptor_heap,
                depth_stencil_buffer,
                dsv_descriptor_size,

                view_port,
                scissor_rect,

                command_allocator,
                command_queue,
                command_list: command_list.clone(),
            },
            RenderingContext::new(command_list),
        )
    }

    pub fn begin_render(&mut self) -> Result<()> {
        unsafe {
            self.current_back_buffer_index = self.swap_chain.GetCurrentBackBufferIndex() as usize;

            self.command_allocator.Reset()?;
            self.command_list.Reset(&self.command_allocator, None)?;

            transition_resource_barrier(
                &self.command_list,
                &self.render_targets[self.current_back_buffer_index],
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );

            // set view port and scissor rect
            self.command_list.RSSetViewports(&[self.view_port]);
            self.command_list.RSSetScissorRects(&[self.scissor_rect]);

            // offset rtv to current back buffer
            let mut rtv_cpu_descriptor_handle = self.rtv_descriptor_heap.GetCPUDescriptorHandleForHeapStart();
            rtv_cpu_descriptor_handle.ptr += self.current_back_buffer_index * self.rtv_descriptor_size;

            // clear current rener target
            let clear_color: [f32; 4] = [0.5, 0.5, 0.5, 1.0];
            self.command_list.ClearRenderTargetView(rtv_cpu_descriptor_handle, clear_color.as_ptr(), &[]);
            self.command_list.ClearDepthStencilView(
                self.dsv_descriptor_heap.GetCPUDescriptorHandleForHeapStart(),
                D3D12_CLEAR_FLAG_DEPTH,
                1.0,
                0,
                &[],
            )
        }

        Ok(())
    }

    pub fn end_render(&mut self) -> Result<()> {
        unsafe {
            transition_resource_barrier(
                &self.command_list,
                &self.render_targets[self.current_back_buffer_index],
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PRESENT,
            );

            self.wait_for_previous_frame()?;

            self.swap_chain.Present(1, 0)?;
        }

        Ok(())
    }

    fn wait_for_previous_frame(&mut self) -> Result<()> {
        unsafe {
            self.command_list.Close()?;

            let command_list: ID3D12CommandList = ID3D12CommandList::from(&self.command_list);
            self.command_queue.ExecuteCommandLists(&[Some(command_list)]);

            self.fence_value += 1;
            self.command_queue.Signal(&self.fence, self.fence_value)?;

            if self.fence.GetCompletedValue() < self.fence_value {
                self.fence.SetEventOnCompletion(self.fence_value, self.fence_event)?;

                WaitForSingleObject(self.fence_event, INFINITE);
            }
        }

        Ok(())
    }

    fn create_command_resource(
        device: &ID3D12Device,
    ) -> Result<(ID3D12CommandAllocator, ID3D12CommandQueue, ID3D12GraphicsCommandList)> {
        // create command allocator
        let command_allocator: ID3D12CommandAllocator =
            unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }?;

        // create command queue
        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
        };
        let command_queue: ID3D12CommandQueue = unsafe { device.CreateCommandQueue(&queue_desc) }?;

        // create command list
        let command_list: ID3D12GraphicsCommandList =
            unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &command_allocator, None) }?;
        unsafe { command_list.Close() }?;

        Ok((command_allocator, command_queue, command_list))
    }

    fn create_swap_chain(
        hwnd: HWND, dxgi_factory: &IDXGIFactory4, command_queue: &ID3D12CommandQueue, window_width: u32,
        window_height: u32,
    ) -> Result<IDXGISwapChain3> {
        let swap_chain_desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: window_width,
            Height: window_height,
            BufferCount: BACK_BUFFER_COUNT as u32,
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            ..Default::default()
        };

        let swap_chain: IDXGISwapChain3 = unsafe {
            dxgi_factory.CreateSwapChainForHwnd(command_queue, hwnd, &swap_chain_desc, std::ptr::null(), None)
        }?
        .cast()?;

        Ok(swap_chain)
    }

    fn create_rtv_descriptor_heap(device: &ID3D12Device) -> Result<ID3D12DescriptorHeap> {
        let descriptor_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
            NumDescriptors: BACK_BUFFER_COUNT as u32,
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        };

        unsafe { device.CreateDescriptorHeap(&descriptor_heap_desc) }
    }

    fn create_render_target_views(
        device: &ID3D12Device, swap_chain: &IDXGISwapChain3, mut cpu_descriptor_handle: D3D12_CPU_DESCRIPTOR_HANDLE,
    ) -> Result<([ID3D12Resource; BACK_BUFFER_COUNT], usize)> {
        let rtv_descriptor_size: usize =
            unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) } as usize;

        let render_targets: [ID3D12Resource; BACK_BUFFER_COUNT] =
            array_init::try_array_init(|i: usize| -> Result<ID3D12Resource> {
                let render_target: ID3D12Resource = unsafe { swap_chain.GetBuffer(i as u32) }?;
                unsafe { device.CreateRenderTargetView(&render_target, std::ptr::null(), cpu_descriptor_handle) };
                cpu_descriptor_handle.ptr += rtv_descriptor_size;

                if cfg!(debug_assertions) {
                    let name = format!("Default_RenderTarget_{}", i);
                    let mut name_vec = name.encode_utf16().collect::<Vec<u16>>();
                    name_vec.push('\0' as u16);

                    unsafe { render_target.SetName(PCWSTR(name_vec.as_ptr())) }.unwrap();
                }

                Ok(render_target)
            })?;

        Ok((render_targets, rtv_descriptor_size))
    }

    fn create_dsv_descriptor_heap(device: &ID3D12Device) -> Result<ID3D12DescriptorHeap> {
        let descriptor_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
            NumDescriptors: 1,
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
        };

        unsafe { device.CreateDescriptorHeap(&descriptor_heap_desc) }
    }

    fn create_depth_stencil_buffer_and_view(
        device: &ID3D12Device, window_width: u32, window_height: u32,
        cpu_descriptor_handle: D3D12_CPU_DESCRIPTOR_HANDLE,
    ) -> Result<(ID3D12Resource, usize)> {
        let dsv_descriptor_size =
            unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV) } as usize;

        let heap_properties = D3D12_HEAP_PROPERTIES {
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 0,
            Type: D3D12_HEAP_TYPE_DEFAULT,
            VisibleNodeMask: 0,
        };

        let resource_desc = D3D12_RESOURCE_DESC {
            Alignment: 0,
            DepthOrArraySize: 1,
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            Format: DXGI_FORMAT_D32_FLOAT,
            Width: window_width as u64,
            Height: window_height as u32,
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        };

        let clear_value = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_D32_FLOAT,
            Anonymous: D3D12_CLEAR_VALUE_0 {
                DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 },
            },
        };

        let mut resource: Option<ID3D12Resource> = None;
        unsafe {
            device.CreateCommittedResource(
                &heap_properties,
                D3D12_HEAP_FLAG_NONE,
                &resource_desc,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                &clear_value,
                &mut resource,
            )
        }
        .unwrap();

        let resource = resource.unwrap();
        
        if cfg!(debug_assertions) {
            unsafe { resource.SetName(PCWSTR(utf16!("Default_DepthStencilBuffer").as_ptr())) }.unwrap();
        }

        unsafe { device.CreateDepthStencilView(&resource, std::ptr::null(), cpu_descriptor_handle) };

        Ok((resource, dsv_descriptor_size))
    }

    fn create_fence(device: &ID3D12Device) -> Result<(u64, ID3D12Fence, HANDLE)> {
        let mut fence_value: u64 = 0;
        let fence: ID3D12Fence = unsafe { device.CreateFence(fence_value, D3D12_FENCE_FLAG_NONE) }?;
        fence_value += 1;

        let fence_event = unsafe { CreateEventA(std::ptr::null(), false, false, None) }?;

        if fence_event.0 == 0 {
            panic!()
        }

        Ok((fence_value, fence, fence_event))
    }

    fn create_viewport_and_scissor_rect(window_width: u32, window_height: u32) -> (D3D12_VIEWPORT, RECT) {
        let view_port = D3D12_VIEWPORT {
            Width: window_width as f32,
            Height: window_height as f32,
            MinDepth: D3D12_MIN_DEPTH,
            MaxDepth: D3D12_MAX_DEPTH,
            TopLeftX: 0 as f32,
            TopLeftY: 0 as f32,
        };

        let scissor_rect = RECT {
            right: window_width as i32,
            bottom: window_height as i32,
            left: 0,
            top: 0,
        };

        (view_port, scissor_rect)
    }
}

#[inline]
fn transition_resource_barrier(
    command_list: &ID3D12GraphicsCommandList, resource: &ID3D12Resource, state_before: D3D12_RESOURCE_STATES,
    state_after: D3D12_RESOURCE_STATES,
) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: Some(resource.clone()),
                StateBefore: state_before,
                StateAfter: state_after,
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            }),
        },
    };

    unsafe {
        command_list.ResourceBarrier(&[barrier.clone()]);
        let _ = std::mem::ManuallyDrop::into_inner(barrier.Anonymous.Transition);
    };
}
