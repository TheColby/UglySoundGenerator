use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_usg")
}

fn temp_dir(label: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("usg_verify_{label}_{now}"));
    fs::create_dir_all(&dir).expect("mkdir");
    dir
}

fn run_ok(args: &[&str]) {
    let out = Command::new(bin())
        .args(args)
        .output()
        .expect("command run");
    assert!(
        out.status.success(),
        "command failed: {}\nstderr:\n{}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn render_is_reproducible_with_same_seed_and_randomness_controls() {
    let dir = temp_dir("render_repro");
    let a = dir.join("a.wav");
    let b = dir.join("b.wav");

    let common = [
        "render",
        "--duration",
        "0.12",
        "--style",
        "catastrophic",
        "--seed",
        "424242",
        "--randomness",
        "0.65",
        "--timing-randomness",
        "0.80",
        "--spectral-randomness",
        "0.70",
        "--amplitude-randomness",
        "0.55",
        "--density-randomness",
        "0.90",
        "--seed-offset",
        "17",
        "--seed-salt",
        "91",
        "--seed-rerolls",
        "2",
    ];

    let mut first: Vec<&str> = common.into();
    first.extend(["--output", a.to_str().expect("path")]);
    run_ok(&first);

    let common = [
        "render",
        "--duration",
        "0.12",
        "--style",
        "catastrophic",
        "--seed",
        "424242",
        "--randomness",
        "0.65",
        "--timing-randomness",
        "0.80",
        "--spectral-randomness",
        "0.70",
        "--amplitude-randomness",
        "0.55",
        "--density-randomness",
        "0.90",
        "--seed-offset",
        "17",
        "--seed-salt",
        "91",
        "--seed-rerolls",
        "2",
    ];
    let mut second: Vec<&str> = common.into();
    second.extend(["--output", b.to_str().expect("path")]);
    run_ok(&second);

    let a_bytes = fs::read(&a).expect("read a");
    let b_bytes = fs::read(&b).expect("read b");
    assert_eq!(a_bytes, b_bytes, "same seed/settings should be byte-stable");
}

#[test]
fn analyze_timeline_json_is_monotonic_and_bounded() {
    let dir = temp_dir("timeline_contract");
    let wav = dir.join("input.wav");
    let timeline = dir.join("timeline.json");

    run_ok(&[
        "render",
        "--output",
        wav.to_str().expect("wav path"),
        "--duration",
        "0.18",
        "--style",
        "glitch",
        "--seed",
        "777",
    ]);

    run_ok(&[
        "analyze",
        wav.to_str().expect("wav path"),
        "--timeline",
        "--timeline-format",
        "json",
        "--timeline-output",
        timeline.to_str().expect("timeline path"),
    ]);

    let frames: Vec<Value> =
        serde_json::from_str(&fs::read_to_string(&timeline).expect("timeline read"))
            .expect("timeline json");
    assert!(frames.len() >= 2, "expected multiple timeline frames");

    let mut prev_time = -1.0_f64;
    for frame in &frames {
        let time_s = frame["time_s"].as_f64().expect("time_s");
        let colbys = frame["colbys"].as_f64().expect("colbys");
        let clipped_pct = frame["clipped_pct"].as_f64().expect("clipped_pct");
        let harshness_ratio = frame["harshness_ratio"].as_f64().expect("harshness_ratio");
        let zcr = frame["zero_crossing_rate"]
            .as_f64()
            .expect("zero_crossing_rate");

        assert!(time_s >= 0.0, "timeline times must be non-negative");
        assert!(time_s >= prev_time, "timeline times must be monotonic");
        assert!((-1000.0..=1000.0).contains(&colbys), "colbys out of range");
        assert!(clipped_pct.is_finite(), "clipped_pct must be finite");
        assert!(
            harshness_ratio.is_finite(),
            "harshness_ratio must be finite"
        );
        assert!(zcr.is_finite(), "zero_crossing_rate must be finite");
        prev_time = time_s;
    }

    let first_time = frames[0]["time_s"].as_f64().expect("first time");
    assert!(
        first_time.abs() < 1e-9,
        "first timeline frame should start at t=0"
    );
}

#[test]
fn speech_pack_summary_has_unique_profiles_and_sorted_ranking() {
    let dir = temp_dir("speech_pack_contract");
    let out_dir = dir.join("speech_pack");

    run_ok(&[
        "speech-pack",
        "--text",
        "VERIFICATION LIKES TERRIBLE ROBOT VOICES",
        "--out-dir",
        out_dir.to_str().expect("out dir"),
        "--rank-by",
        "balanced",
    ]);

    let summary_path = out_dir.join("summary.json");
    let summary: Value =
        serde_json::from_str(&fs::read_to_string(&summary_path).expect("summary read"))
            .expect("summary json");

    assert_eq!(
        summary["profiles_rendered"]
            .as_u64()
            .expect("profiles_rendered"),
        8
    );

    let entries = summary["entries"].as_array().expect("entries");
    let ranking = summary["ranking"].as_array().expect("ranking");
    assert_eq!(entries.len(), 8, "expected one entry per speech profile");
    assert_eq!(
        ranking.len(),
        8,
        "expected one ranking row per speech profile"
    );

    let mut seen = HashSet::new();
    for entry in entries {
        let profile = entry["profile"].as_str().expect("profile");
        assert!(
            seen.insert(profile.to_string()),
            "duplicate profile {profile}"
        );
    }

    let mut prev_score = f64::INFINITY;
    for (idx, row) in ranking.iter().enumerate() {
        let rank = row["rank"].as_u64().expect("rank") as usize;
        let score = row["rank_score"].as_f64().expect("rank_score");
        assert_eq!(rank, idx + 1, "rank ordering should be sequential");
        assert!(
            score <= prev_score + 1e-9,
            "rank_score should be sorted descending"
        );
        prev_score = score;
    }
}

#[test]
fn mutate_summary_delta_matches_base_plus_result_score() {
    let dir = temp_dir("mutate_contract");
    let input = dir.join("input.wav");
    let out_dir = dir.join("mutate_out");

    run_ok(&[
        "render",
        "--output",
        input.to_str().expect("input path"),
        "--duration",
        "0.12",
        "--style",
        "hum",
        "--seed",
        "19",
    ]);

    run_ok(&[
        "mutate",
        input.to_str().expect("input path"),
        "--count",
        "4",
        "--out-dir",
        out_dir.to_str().expect("out dir"),
        "--seed",
        "1234",
    ]);

    let summary_path = out_dir.join("mutate_summary.json");
    let summary: Value =
        serde_json::from_str(&fs::read_to_string(&summary_path).expect("summary read"))
            .expect("summary json");

    let base = summary["base_colbys"].as_f64().expect("base_colbys");
    let entries = summary["entries"].as_array().expect("entries");
    assert_eq!(entries.len(), 4, "expected requested mutation count");

    for entry in entries {
        let colbys = entry["colbys"].as_f64().expect("colbys");
        let ugly_delta = entry["ugly_delta"].as_f64().expect("ugly_delta");
        assert!(
            (ugly_delta - (colbys - base)).abs() < 1e-6,
            "ugly_delta must equal colbys - base_colbys"
        );
    }
}
