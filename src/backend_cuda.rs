#![cfg(feature = "cuda")]

use anyhow::{Result, anyhow};
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use crate::Style;

const CUDA_KERNEL_SRC: &str = r#"
#define PI_F 3.14159265358979323846f

__device__ unsigned long long mix64(unsigned long long x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

__device__ float fract_usg(float x) {
    return x - floorf(x);
}

__device__ float saw_usg(float freq, float t) {
    float phase = fract_usg(freq * t);
    return 2.0f * phase - 1.0f;
}

__device__ float square_usg(float freq, float t) {
    return fract_usg(freq * t) < 0.5f ? 1.0f : -1.0f;
}

__device__ float triangle_usg(float freq, float t) {
    return fabsf(2.0f * fract_usg(freq * t) - 1.0f) * 2.0f - 1.0f;
}

__device__ float hash_unit(unsigned long long seed, int idx, unsigned long long salt) {
    unsigned long long x = mix64(seed ^ ((unsigned long long)(idx + 1) * salt));
    return ((float)(x & 0x00ffffffULL) / 8388607.5f) - 1.0f;
}

__device__ float pulse_env(float t, float rate, unsigned long long seed, unsigned long long salt, float sharpness) {
    float jitter = 0.5f * (hash_unit(seed, (int)floorf(t * rate), salt) + 1.0f);
    float phase = fract_usg(t * rate + jitter);
    float env = fmaxf(0.0f, 1.0f - phase * sharpness);
    return env * env;
}

__device__ void style_params(int style_id, float* hiss_amp, float* click_prob, float* click_amp, float* glitch_prob, float* glitch_amp, float* bit_depth, float* style_drive) {
    switch (style_id) {
        case 0: *hiss_amp = 0.22f; *click_prob = 0.0015f; *click_amp = 1.4f; *glitch_prob = 0.0020f; *glitch_amp = 0.75f; *bit_depth = 6.0f; *style_drive = 1.35f; break;
        case 1: *hiss_amp = 0.16f; *click_prob = 0.0012f; *click_amp = 1.2f; *glitch_prob = 0.0015f; *glitch_amp = 0.55f; *bit_depth = 4.0f; *style_drive = 1.3f; break;
        case 2: *hiss_amp = 0.25f; *click_prob = 0.0017f; *click_amp = 1.5f; *glitch_prob = 0.0023f; *glitch_amp = 0.9f; *bit_depth = 5.0f; *style_drive = 1.45f; break;
        case 3: *hiss_amp = 0.09f; *click_prob = 0.0030f; *click_amp = 1.4f; *glitch_prob = 0.0050f; *glitch_amp = 1.2f; *bit_depth = 3.0f; *style_drive = 1.55f; break;
        case 4: *hiss_amp = 0.06f; *click_prob = 0.0060f; *click_amp = 1.8f; *glitch_prob = 0.0010f; *glitch_amp = 0.4f; *bit_depth = 7.0f; *style_drive = 1.25f; break;
        case 5: *hiss_amp = 0.12f; *click_prob = 0.0010f; *click_amp = 0.9f; *glitch_prob = 0.0012f; *glitch_amp = 0.5f; *bit_depth = 6.0f; *style_drive = 1.5f; break;
        case 6: *hiss_amp = 0.18f; *click_prob = 0.0015f; *click_amp = 0.8f; *glitch_prob = 0.0009f; *glitch_amp = 0.4f; *bit_depth = 6.0f; *style_drive = 1.3f; break;
        case 7: *hiss_amp = 0.04f; *click_prob = 0.0003f; *click_amp = 0.5f; *glitch_prob = 0.0005f; *glitch_amp = 0.25f; *bit_depth = 9.0f; *style_drive = 1.1f; break;
        case 8: *hiss_amp = 0.10f; *click_prob = 0.0011f; *click_amp = 1.1f; *glitch_prob = 0.0013f; *glitch_amp = 0.55f; *bit_depth = 5.0f; *style_drive = 1.7f; break;
        case 9: *hiss_amp = 0.08f; *click_prob = 0.0038f; *click_amp = 1.6f; *glitch_prob = 0.0018f; *glitch_amp = 0.65f; *bit_depth = 6.0f; *style_drive = 1.45f; break;
        case 10:*hiss_amp = 0.28f; *click_prob = 0.0032f; *click_amp = 1.9f; *glitch_prob = 0.0038f; *glitch_amp = 1.3f; *bit_depth = 3.0f; *style_drive = 1.95f; break;
        case 11:*hiss_amp = 0.14f; *click_prob = 0.0022f; *click_amp = 1.2f; *glitch_prob = 0.0028f; *glitch_amp = 0.95f; *bit_depth = 4.0f; *style_drive = 1.55f; break;
        case 12:*hiss_amp = 0.07f; *click_prob = 0.0010f; *click_amp = 0.9f; *glitch_prob = 0.0011f; *glitch_amp = 0.45f; *bit_depth = 7.0f; *style_drive = 1.2f; break;
        case 13:*hiss_amp = 0.34f; *click_prob = 0.0065f; *click_amp = 2.2f; *glitch_prob = 0.0072f; *glitch_amp = 1.6f; *bit_depth = 2.0f; *style_drive = 2.35f; break;
        default:*hiss_amp = 0.20f; *click_prob = 0.0024f; *click_amp = 1.4f; *glitch_prob = 0.0027f; *glitch_amp = 1.0f; *bit_depth = 5.0f; *style_drive = 1.5f; break;
    }
}

__device__ float synth_base(int style_id, int idx, float t, float sample_rate, unsigned long long seed) {
    float wobble = sinf(2.0f * PI_F * 0.27f * t) * 4.0f;
    float noise = hash_unit(seed, idx, 0x9e3779b97f4a7c15ULL);
    float hold = hash_unit(seed, idx / 128, 0xbf58476d1ce4e5b9ULL);
    float pulse_pop = pulse_env(t, 8.0f, seed, 0x94d049bb133111ebULL, 10.0f);
    float pulse_spank = pulse_env(t, 13.0f, seed, 0x243f6a8885a308d3ULL, 14.0f);
    int lucky_mode = ((int)(fabsf(hash_unit(seed, idx / (int)fmaxf(1.0f, sample_rate / 5.0f), 0x123456789abcdef0ULL)) * 1000.0f)) % 6;

    switch (style_id) {
        case 0: {
            float saw_a = saw_usg(90.0f + wobble + 55.0f * sinf(2.0f * PI_F * 1.7f * t), t);
            float sq_b = square_usg(180.0f + 40.0f * sinf(2.0f * PI_F * 3.1f * t), t);
            return 0.65f * saw_a + 0.45f * sq_b;
        }
        case 1: {
            float tri = triangle_usg(150.0f + 90.0f * sinf(2.0f * PI_F * 11.0f * t), t);
            float ring = sinf(2.0f * PI_F * 400.0f * t) * sinf(2.0f * PI_F * 1250.0f * t);
            return 0.5f * tri + 0.8f * ring;
        }
        case 2: {
            float fm = sinf(2.0f * PI_F * (110.0f + 550.0f * sinf(2.0f * PI_F * 0.9f * t)) * t);
            float sub = saw_usg(55.0f + 20.0f * sinf(2.0f * PI_F * 0.2f * t), t);
            return 0.75f * fm + 0.35f * sub;
        }
        case 3: {
            float staircase = ((idx / 6) % 48) / 24.0f - 1.0f;
            float fold = copysignf(1.0f, sinf(2.0f * PI_F * (60.0f + 420.0f * fabsf(sinf(2.0f * PI_F * 0.4f * t))) * t));
            return 0.65f * staircase + 0.55f * fold;
        }
        case 4: {
            float body = sinf(2.0f * PI_F * (80.0f + 2400.0f * pulse_pop) * t);
            return 0.9f * pulse_pop * body + 0.35f * noise * pulse_pop;
        }
        case 5: {
            float f = 96.0f + 20.0f * sinf(2.0f * PI_F * 0.3f * t);
            float harmonics = 0.0f;
            for (int h = 1; h <= 9; ++h) {
                harmonics += saw_usg(f * h, t) / (float)h;
            }
            return 0.75f * harmonics + 0.25f * square_usg(f * 2.0f, t);
        }
        case 6: {
            float friction = noise * (0.45f + 0.55f * fabsf(sinf(2.0f * PI_F * 5.0f * t)));
            float scrape = saw_usg(35.0f + 12.0f * sinf(2.0f * PI_F * 0.7f * t), t) * 0.25f;
            return 1.1f * friction + scrape;
        }
        case 7:
            return 0.8f * sinf(2.0f * PI_F * 60.0f * t) + 0.32f * sinf(2.0f * PI_F * 120.0f * t) + 0.15f * sinf(2.0f * PI_F * 180.0f * t) + 0.05f * sinf(2.0f * PI_F * 7.0f * t);
        case 8: {
            float dry = 0.65f * sinf(2.0f * PI_F * 140.0f * t) + 0.45f * saw_usg(280.0f, t) + 0.25f * square_usg(35.0f, t);
            return tanhf(dry * 3.8f);
        }
        case 9: {
            float snap = sinf(2.0f * PI_F * (190.0f + 4200.0f * pulse_spank) * t);
            float tail = saw_usg(70.0f, t) * pulse_spank * 0.4f;
            return 1.1f * pulse_spank * snap + tail;
        }
        case 10: {
            float fm = sinf(2.0f * PI_F * (90.0f + 1400.0f * sinf(2.0f * PI_F * 2.4f * t)) * t);
            float sq = square_usg(43.0f, t);
            return tanhf((fm + 0.8f * sq + 0.4f * noise) * 4.2f);
        }
        case 11: {
            float held = 0.85f * hold + 0.45f * saw_usg(120.0f + 65.0f * sinf(2.0f * PI_F * 0.2f * t), t);
            float ghost = square_usg(40.0f + 170.0f * fabsf(held), t) * 0.25f;
            return held + ghost;
        }
        case 12: {
            float gate = sinf(2.0f * PI_F * 2.8f * t) >= 0.0f ? 1.0f : -1.0f;
            float chirp_freq = 250.0f + 2400.0f * fract_usg(t * 1.6f);
            float chirp = sinf(2.0f * PI_F * chirp_freq * t);
            return 0.7f * chirp * gate + 0.35f * triangle_usg(90.0f, t);
        }
        case 13: {
            float staircase = ((idx / 3) % 24) / 12.0f - 1.0f;
            float held = hash_unit(seed, idx / 21, 0x517cc1b727220a95ULL);
            float env = 0.45f + 0.55f * fabsf(sinf(2.0f * PI_F * 0.61f * t));
            float fm = sinf(2.0f * PI_F * (120.0f + 1800.0f * sinf(2.0f * PI_F * 3.7f * t) + 420.0f * fabsf(held)) * t);
            float shriek = saw_usg(420.0f + 4200.0f * (0.5f + 0.5f * sinf(2.0f * PI_F * 0.43f * t)) + 600.0f * env, t);
            float sub = square_usg(27.0f + 90.0f * fabsf(held), t);
            float raw = 0.55f * fm + 0.45f * staircase + 0.45f * shriek + 0.35f * sub + 0.6f * held + 0.5f * noise;
            return (sinf(raw * 5.6f) + tanhf(raw * 4.4f) + 0.55f * copysignf(1.0f, raw)) / 1.9f;
        }
        case 14:
        default: {
            switch (lucky_mode) {
                case 0: return 0.8f * saw_usg(80.0f + 20.0f * sinf(2.0f * PI_F * 1.5f * t), t);
                case 1: return 0.7f * square_usg(42.0f, t) + 0.4f * triangle_usg(620.0f, t);
                case 2: return 0.6f * sinf(2.0f * PI_F * (220.0f + 600.0f * sinf(2.0f * PI_F * 0.8f * t)) * t);
                case 3: return 0.5f * saw_usg(140.0f, t) + 0.5f * noise;
                case 4: return 0.9f * (((idx / 8) % 32) / 16.0f - 1.0f);
                default:return 0.6f * triangle_usg(300.0f + 120.0f * sinf(2.0f * PI_F * 4.0f * t), t);
            }
        }
    }
}

extern "C" __global__ void usg_render_style(
    float* out_samples,
    int n,
    float sample_rate,
    int style_id,
    float gain,
    unsigned long long seed,
    float drive,
    float crush_levels,
    float crush_mix
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    float hiss_amp, click_prob, click_amp, glitch_prob, glitch_amp, bit_depth, style_drive;
    style_params(style_id, &hiss_amp, &click_prob, &click_amp, &glitch_prob, &glitch_amp, &bit_depth, &style_drive);

    float t = idx / sample_rate;
    float noise = hash_unit(seed, idx, 0x9e3779b97f4a7c15ULL);
    float click_gate = 0.5f * (hash_unit(seed, idx / 17, 0xbf58476d1ce4e5b9ULL) + 1.0f);
    float glitch_gate = 0.5f * (hash_unit(seed, idx / 96, 0x94d049bb133111ebULL) + 1.0f);
    float click = click_gate < click_prob * 32.0f ? noise * click_amp : 0.0f;
    float held = ((idx / 24) % 32) / 31.0f;
    float glitch = glitch_gate < glitch_prob * 48.0f ? (held * 2.0f - 1.0f) * glitch_amp : 0.0f;
    float base = synth_base(style_id, idx, t, sample_rate, seed);
    float raw = base + noise * hiss_amp + click + glitch;
    float levels = exp2f(bit_depth);
    float crushed = roundf(raw * levels) / levels;
    float v = tanhf(crushed * gain * style_drive);
    v = tanhf(v * drive);
    if (crush_levels > 1.0f && crush_mix > 0.0f) {
        float crushed_post = roundf(v * crush_levels) / crush_levels;
        v = v * (1.0f - crush_mix) + crushed_post * crush_mix;
    }
    if (v > 1.0f) v = 1.0f;
    if (v < -1.0f) v = -1.0f;
    out_samples[idx] = v;
}

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

fn style_id(style: Style) -> i32 {
    match style {
        Style::Harsh => 0,
        Style::Digital => 1,
        Style::Meltdown => 2,
        Style::Glitch => 3,
        Style::Pop => 4,
        Style::Buzz => 5,
        Style::Rub => 6,
        Style::Hum => 7,
        Style::Distort => 8,
        Style::Spank => 9,
        Style::Punish => 10,
        Style::Steal => 11,
        Style::Wink => 12,
        Style::Catastrophic => 13,
        Style::Lucky => 14,
    }
}

pub fn available() -> bool {
    CudaContext::new(0).is_ok()
}

pub fn availability_detail() -> String {
    match CudaContext::new(0) {
        Ok(_) => "CUDA device/context opened successfully".to_string(),
        Err(err) => format!("CUDA unavailable: {err:?}"),
    }
}

pub fn render_style(
    style: Style,
    frames: usize,
    sample_rate: f64,
    gain: f64,
    seed: u64,
    drive: f64,
    crush_bits: f64,
    crush_mix: f64,
) -> Result<Vec<f64>> {
    if frames == 0 {
        return Ok(Vec::new());
    }
    let drive = drive.clamp(0.1, 16.0) as f32;
    let crush_bits = crush_bits.clamp(0.0, 24.0);
    let crush_levels = if crush_bits > 0.0 {
        (2.0_f64).powf(crush_bits) as f32
    } else {
        0.0_f32
    };
    let crush_mix = crush_mix.clamp(0.0, 1.0) as f32;
    let gain = gain.clamp(0.0, 1.0) as f32;
    let style_id = style_id(style);

    let ctx = CudaContext::new(0).map_err(|e| anyhow!("failed to open CUDA device: {e:?}"))?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(CUDA_KERNEL_SRC).map_err(|e| anyhow!("failed to compile PTX: {e:?}"))?;
    let module = ctx
        .load_module(ptx)
        .map_err(|e| anyhow!("failed to load PTX module: {e:?}"))?;
    let func = module
        .load_function("usg_render_style")
        .map_err(|e| anyhow!("failed to get CUDA kernel usg_render_style: {e:?}"))?;

    let mut d_out = stream
        .alloc_zeros::<f32>(frames)
        .map_err(|e| anyhow!("failed to allocate CUDA render output: {e:?}"))?;
    let cfg = LaunchConfig::for_num_elems(frames as u32);
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&mut d_out)
            .arg(&(frames as i32))
            .arg(&(sample_rate as f32))
            .arg(&style_id)
            .arg(&gain)
            .arg(&seed)
            .arg(&drive)
            .arg(&crush_levels)
            .arg(&crush_mix)
            .launch(cfg)
            .map_err(|e| anyhow!("CUDA render kernel launch failed: {e:?}"))?;
    }
    let out = stream
        .memcpy_dtov(&d_out)
        .map_err(|e| anyhow!("failed to download CUDA render output: {e:?}"))?;
    Ok(out.into_iter().map(|v| v as f64).collect())
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
