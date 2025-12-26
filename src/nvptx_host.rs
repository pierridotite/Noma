//! NVPTX host stubs and CUDA Driver API wrappers (feature-gated)

#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::ffi::{c_void, CString};

    // Minimal CUDA Driver API types
    type CUdevice = i32;
    type CUcontext = *mut c_void;
    type CUmodule = *mut c_void;
    type CUfunction = *mut c_void;
    type CUdeviceptr = u64;
    type CUresult = i32;

    const CUDA_SUCCESS: CUresult = 0;

    #[link(name = "cuda")]
    extern "C" {
        fn cuInit(flags: u32) -> CUresult;
        fn cuDeviceGet(device: *mut CUdevice, ordinal: i32) -> CUresult;
        fn cuCtxCreate_v2(ctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult;
        fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
        fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
        fn cuModuleUnload(module: CUmodule) -> CUresult;
        fn cuModuleGetFunction(hfunc: *mut CUfunction, module: CUmodule, name: *const i8) -> CUresult;
        fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
        fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
        fn cuMemcpyHtoD_v2(dstDevice: CUdeviceptr, srcHost: *const c_void, ByteCount: usize) -> CUresult;
        fn cuMemcpyDtoH_v2(dstHost: *mut c_void, srcDevice: CUdeviceptr, ByteCount: usize) -> CUresult;
        fn cuLaunchKernel(f: CUfunction,
                          gridDimX: u32, gridDimY: u32, gridDimZ: u32,
                          blockDimX: u32, blockDimY: u32, blockDimZ: u32,
                          sharedMemBytes: u32,
                          hStream: *mut c_void,
                          kernelParams: *mut *mut c_void,
                          extra: *mut *mut c_void) -> CUresult;
        fn cuCtxSynchronize() -> CUresult;
    }

    fn check(res: CUresult, msg: &str) -> Result<(), String> {
        if res == CUDA_SUCCESS { Ok(()) } else { Err(format!("CUDA error {}: {}", res, msg)) }
    }

    pub fn run_elementwise_kernel(ptx: &str, func_name: &str, n_elems: u32) -> Result<Vec<f64>, String> {
        unsafe {
            check(cuInit(0), "cuInit")?;
            let mut dev: CUdevice = 0;
            check(cuDeviceGet(&mut dev as *mut CUdevice, 0), "cuDeviceGet")?;
            let mut ctx: CUcontext = std::ptr::null_mut();
            check(cuCtxCreate_v2(&mut ctx as *mut CUcontext, 0, dev), "cuCtxCreate")?;

            let c_ptx = CString::new(ptx).map_err(|_| "PTX contains interior NUL".to_string())?;
            let mut module: CUmodule = std::ptr::null_mut();
            check(cuModuleLoadData(&mut module as *mut CUmodule, c_ptx.as_ptr() as *const c_void), "cuModuleLoadData")?;

            let c_name = CString::new(func_name).unwrap();
            let mut func: CUfunction = std::ptr::null_mut();
            check(cuModuleGetFunction(&mut func as *mut CUfunction, module, c_name.as_ptr()), "cuModuleGetFunction")?;

            let bytes = (n_elems as usize) * 8;
            let mut d_in: CUdeviceptr = 0;
            let mut d_out: CUdeviceptr = 0;
            check(cuMemAlloc_v2(&mut d_in as *mut CUdeviceptr, bytes), "cuMemAlloc in")?;
            check(cuMemAlloc_v2(&mut d_out as *mut CUdeviceptr, bytes), "cuMemAlloc out")?;

            let h_in: Vec<f64> = vec![1.0; n_elems as usize];
            let mut h_out: Vec<f64> = vec![0.0; n_elems as usize];
            check(cuMemcpyHtoD_v2(d_in, h_in.as_ptr() as *const c_void, bytes), "cuMemcpyHtoD in")?;

            // Params: &d_in, &d_out, &n_elems
            let mut p_in: CUdeviceptr = d_in;
            let mut p_out: CUdeviceptr = d_out;
            let mut p_n: u32 = n_elems;
            let mut params: [*mut c_void; 3] = [
                (&mut p_in as *mut CUdeviceptr) as *mut c_void,
                (&mut p_out as *mut CUdeviceptr) as *mut c_void,
                (&mut p_n as *mut u32) as *mut c_void,
            ];

            let block = 128u32;
            let grid = (n_elems + block - 1) / block;
            check(cuLaunchKernel(
                func,
                grid, 1, 1,
                block, 1, 1,
                0,
                std::ptr::null_mut(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            ), "cuLaunchKernel")?;

            check(cuCtxSynchronize(), "cuCtxSynchronize")?;
            check(cuMemcpyDtoH_v2(h_out.as_mut_ptr() as *mut c_void, d_out, bytes), "cuMemcpyDtoH out")?;

            // Cleanup
            let _ = cuMemFree_v2(d_in);
            let _ = cuMemFree_v2(d_out);
            let _ = cuModuleUnload(module);
            let _ = cuCtxDestroy_v2(ctx);

            Ok(h_out)
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod noop_impl {
    pub fn run_elementwise_kernel(_ptx: &str, _func_name: &str, _n_elems: u32) -> Result<Vec<f64>, String> {
        Err("CUDA feature not enabled; build with --features cuda".to_string())
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::run_elementwise_kernel;
#[cfg(not(feature = "cuda"))]
pub use noop_impl::run_elementwise_kernel;
