#![cfg(all(feature = "metal", target_os = "macos"))]

use std::ffi::c_void;
use std::mem::size_of;

use anyhow::{Result, anyhow};
use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize, NSUInteger};

const METAL_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void usg_post_fx(
    const device float *in_samples [[buffer(0)]],
    device float *out_samples [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant float &drive [[buffer(3)]],
    constant float &crush_levels [[buffer(4)]],
    constant float &crush_mix [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) {
        return;
    }
    float v = in_samples[gid];
    v = tanh(v * drive);
    if (crush_levels > 1.0f && crush_mix > 0.0f) {
        float crushed = round(v * crush_levels) / crush_levels;
        v = v * (1.0f - crush_mix) + crushed * crush_mix;
    }
    if (v > 1.0f) {
        v = 1.0f;
    } else if (v < -1.0f) {
        v = -1.0f;
    }
    out_samples[gid] = v;
}
"#;

pub fn available() -> bool {
    Device::system_default().is_some()
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

    let device = Device::system_default().ok_or_else(|| anyhow!("no Metal device found"))?;
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(METAL_KERNEL_SRC, &options)
        .map_err(|e| anyhow!("failed to compile Metal source: {e}"))?;
    let function = library
        .get_function("usg_post_fx", None)
        .map_err(|e| anyhow!("failed to load usg_post_fx: {e}"))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow!("failed to build Metal pipeline: {e}"))?;

    let in_f32: Vec<f32> = samples.iter().map(|&x| x as f32).collect();
    let sample_bytes = (in_f32.len() * size_of::<f32>()) as u64;
    let in_buffer = device.new_buffer_with_data(
        in_f32.as_ptr() as *const c_void,
        sample_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buffer = device.new_buffer(sample_bytes, MTLResourceOptions::StorageModeShared);
    let n = samples.len() as u32;
    let n_buffer = device.new_buffer_with_data(
        (&n as *const u32) as *const c_void,
        size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let drive_buffer = device.new_buffer_with_data(
        (&drive as *const f32) as *const c_void,
        size_of::<f32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let crush_levels_buffer = device.new_buffer_with_data(
        (&crush_levels as *const f32) as *const c_void,
        size_of::<f32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let crush_mix_buffer = device.new_buffer_with_data(
        (&crush_mix as *const f32) as *const c_void,
        size_of::<f32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&in_buffer), 0);
    encoder.set_buffer(1, Some(&out_buffer), 0);
    encoder.set_buffer(2, Some(&n_buffer), 0);
    encoder.set_buffer(3, Some(&drive_buffer), 0);
    encoder.set_buffer(4, Some(&crush_levels_buffer), 0);
    encoder.set_buffer(5, Some(&crush_mix_buffer), 0);

    let w = pipeline.thread_execution_width() as u64;
    let threads_per_group = MTLSize {
        width: w.max(1),
        height: 1,
        depth: 1,
    };
    let threadgroups = MTLSize {
        width: (samples.len() as u64).div_ceil(threads_per_group.width),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let status = command_buffer.status();
    if status != metal::MTLCommandBufferStatus::Completed {
        return Err(anyhow!(
            "Metal command failed with status {:?}",
            status as NSUInteger
        ));
    }

    let out_ptr = out_buffer.contents() as *const f32;
    let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, samples.len()) };
    for (dst, src) in samples.iter_mut().zip(out_slice.iter().copied()) {
        *dst = src as f64;
    }
    Ok(())
}
