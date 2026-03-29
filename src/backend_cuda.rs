#![cfg(feature = "cuda")]

use anyhow::{Result, anyhow};
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

const CUDA_KERNEL_SRC: &str = r#"
extern "C" __global__ void usg_post_fx(
    const float* in_samples,
    float* out_samples,
    int n,
    float drive,
    float crush_levels,
    float crush_mix
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float v = in_samples[idx];
    v = tanhf(v * drive);
    if (crush_levels > 1.0f && crush_mix > 0.0f) {
        float crushed = roundf(v * crush_levels) / crush_levels;
        v = v * (1.0f - crush_mix) + crushed * crush_mix;
    }
    if (v > 1.0f) v = 1.0f;
    if (v < -1.0f) v = -1.0f;
    out_samples[idx] = v;
}
"#;

pub fn available() -> bool {
    CudaContext::new(0).is_ok()
}

pub fn post_fx_in_place(
    samples: &mut [f64],
    drive: f64,
    crush_bits: f64,
    crush_mix: f64,
    _jobs: usize,
) -> Result<()> {
    if samples.is_empty() {
        return Ok(());
    }

    let drive = drive.clamp(0.1, 16.0) as f32;
    let crush_bits = crush_bits.clamp(0.0, 24.0);
    let crush_levels = if crush_bits > 0.0 {
        (2.0_f64).powf(crush_bits) as f32
    } else {
        0.0_f32
    };
    let crush_mix = crush_mix.clamp(0.0, 1.0) as f32;

    let ctx = CudaContext::new(0).map_err(|e| anyhow!("failed to open CUDA device: {e:?}"))?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(CUDA_KERNEL_SRC).map_err(|e| anyhow!("failed to compile PTX: {e:?}"))?;
    let module = ctx
        .load_module(ptx)
        .map_err(|e| anyhow!("failed to load PTX module: {e:?}"))?;
    let func = module
        .load_function("usg_post_fx")
        .map_err(|e| anyhow!("failed to get CUDA kernel usg_post_fx: {e:?}"))?;

    let in_f32: Vec<f32> = samples.iter().map(|&x| x as f32).collect();
    let d_in = stream
        .memcpy_stod(&in_f32)
        .map_err(|e| anyhow!("failed to upload samples to CUDA: {e:?}"))?;
    let mut d_out = stream
        .alloc_zeros::<f32>(samples.len())
        .map_err(|e| anyhow!("failed to allocate CUDA output: {e:?}"))?;

    let cfg = LaunchConfig::for_num_elems(samples.len() as u32);
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&d_in)
            .arg(&mut d_out)
            .arg(&(samples.len() as i32))
            .arg(&drive)
            .arg(&crush_levels)
            .arg(&crush_mix)
            .launch(cfg)
            .map_err(|e| anyhow!("CUDA kernel launch failed: {e:?}"))?;
    }
    let out = stream
        .memcpy_dtov(&d_out)
        .map_err(|e| anyhow!("failed to download CUDA output: {e:?}"))?;
    for (dst, src) in samples.iter_mut().zip(out.into_iter()) {
        *dst = src as f64;
    }
    Ok(())
}
