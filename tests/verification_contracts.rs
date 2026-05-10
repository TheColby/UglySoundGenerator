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

fn manifest_entry_seeds(manifest: &Value) -> Vec<u64> {
    manifest["entries"]
        .as_array()
        .expect("entries")
        .iter()
        .map(|entry| entry["seed"].as_u64().expect("entry seed"))
        .collect()
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
fn render_random_presets_are_deterministic_and_distinct() {
    let dir = temp_dir("render_random_presets");
    let stable_a = dir.join("stable_a.wav");
    let stable_b = dir.join("stable_b.wav");
    let feral = dir.join("feral.wav");

    let common = [
        "render",
        "--duration",
        "0.12",
        "--sample-rate",
        "22050",
        "--style",
        "glitch",
        "--seed",
        "5150",
    ];

    let mut stable_a_args: Vec<&str> = common.into();
    stable_a_args.extend([
        "--random-preset",
        "stable",
        "--output",
        stable_a.to_str().expect("stable a path"),
    ]);
    run_ok(&stable_a_args);

    let common = [
        "render",
        "--duration",
        "0.12",
        "--sample-rate",
        "22050",
        "--style",
        "glitch",
        "--seed",
        "5150",
    ];
    let mut stable_b_args: Vec<&str> = common.into();
    stable_b_args.extend([
        "--random-preset",
        "stable",
        "--output",
        stable_b.to_str().expect("stable b path"),
    ]);
    run_ok(&stable_b_args);

    let common = [
        "render",
        "--duration",
        "0.12",
        "--sample-rate",
        "22050",
        "--style",
        "glitch",
        "--seed",
        "5150",
    ];
    let mut feral_args: Vec<&str> = common.into();
    feral_args.extend([
        "--random-preset",
        "feral",
        "--output",
        feral.to_str().expect("feral path"),
    ]);
    run_ok(&feral_args);

    let stable_a_bytes = fs::read(&stable_a).expect("read stable a");
    let stable_b_bytes = fs::read(&stable_b).expect("read stable b");
    let feral_bytes = fs::read(&feral).expect("read feral");

    assert_eq!(
        stable_a_bytes, stable_b_bytes,
        "same named preset and seed should be byte-stable"
    );
    assert_ne!(
        stable_a_bytes, feral_bytes,
        "different named randomness presets should resolve to distinct render settings"
    );
}

#[test]
fn piece_manifest_records_reproducible_event_seed_plan() {
    let dir = temp_dir("piece_manifest");
    let wav = dir.join("piece.wav");
    let manifest_path = dir.join("piece_manifest.json");

    run_ok(&[
        "piece",
        "--output",
        wav.to_str().expect("wav path"),
        "--manifest",
        manifest_path.to_str().expect("manifest path"),
        "--duration",
        "0.3",
        "--sample-rate",
        "22050",
        "--styles",
        "glitch,punish",
        "--events-per-second",
        "8",
        "--min-event-duration",
        "0.02",
        "--max-event-duration",
        "0.05",
        "--seed",
        "2024",
        "--seed-rerolls",
        "1",
        "--layer-rerolls",
        "2",
    ]);

    let manifest: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest read"))
            .expect("piece manifest json");

    assert!(
        manifest["base_seed"].is_u64() || manifest["seed"].is_u64(),
        "manifest should record the effective base seed: {manifest:#}"
    );
    assert!(
        manifest["events"].as_array().is_some(),
        "manifest should include event-level reproduction data: {manifest:#}"
    );
    let events = manifest["events"].as_array().expect("events");
    assert!(!events.is_empty(), "piece manifest should include events");
    for event in events {
        assert!(
            event["seed"].is_u64(),
            "event should record seed: {event:#}"
        );
        assert!(
            event["style"].is_string(),
            "event should record style: {event:#}"
        );
        assert!(
            event["start_s"].is_number() || event["start_frame"].is_u64(),
            "event should record placement timing: {event:#}"
        );
    }
}

#[test]
fn marathon_seed_rerolls_are_deterministic_and_change_seed_plan() {
    let dir = temp_dir("marathon_rerolls");
    let first = dir.join("first");
    let second = dir.join("second");
    let baseline = dir.join("baseline");

    let common = [
        "--count",
        "3",
        "--min-duration",
        "0.05",
        "--max-duration",
        "0.05",
        "--sample-rate",
        "22050",
        "--styles",
        "glitch,punish",
        "--seed",
        "777",
        "--seed-offset",
        "5",
        "--seed-salt",
        "3",
        "--seed-rerolls",
        "2",
    ];

    let mut first_args = vec!["marathon", "--out-dir", first.to_str().expect("first dir")];
    first_args.extend(common);
    run_ok(&first_args);

    let mut second_args = vec![
        "marathon",
        "--out-dir",
        second.to_str().expect("second dir"),
    ];
    second_args.extend(common);
    run_ok(&second_args);

    let baseline_args = vec![
        "marathon",
        "--out-dir",
        baseline.to_str().expect("baseline dir"),
        "--count",
        "3",
        "--min-duration",
        "0.05",
        "--max-duration",
        "0.05",
        "--sample-rate",
        "22050",
        "--styles",
        "glitch,punish",
        "--seed",
        "777",
        "--seed-offset",
        "5",
        "--seed-salt",
        "3",
        "--seed-rerolls",
        "0",
    ];
    run_ok(&baseline_args);

    let first_manifest: Value =
        serde_json::from_str(&fs::read_to_string(first.join("manifest.json")).expect("first read"))
            .expect("first manifest");
    let second_manifest: Value = serde_json::from_str(
        &fs::read_to_string(second.join("manifest.json")).expect("second read"),
    )
    .expect("second manifest");
    let baseline_manifest: Value = serde_json::from_str(
        &fs::read_to_string(baseline.join("manifest.json")).expect("baseline read"),
    )
    .expect("baseline manifest");

    assert_eq!(
        first_manifest["base_seed"], second_manifest["base_seed"],
        "same seed reroll controls should produce the same manifest base seed"
    );
    assert_eq!(
        manifest_entry_seeds(&first_manifest),
        manifest_entry_seeds(&second_manifest),
        "same seed reroll controls should produce the same entry seed plan"
    );
    assert_ne!(
        first_manifest["base_seed"], baseline_manifest["base_seed"],
        "changing seed-rerolls should change the derived base seed"
    );
    assert_ne!(
        manifest_entry_seeds(&first_manifest),
        manifest_entry_seeds(&baseline_manifest),
        "changing seed-rerolls should change the entry seed plan"
    );
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

    let profiles_rendered = summary["profiles_rendered"]
        .as_u64()
        .expect("profiles_rendered") as usize;
    assert!(
        profiles_rendered > 8,
        "v0.5 speech packs should include additional chip profiles beyond the v0.4 baseline"
    );

    let entries = summary["entries"].as_array().expect("entries");
    let ranking = summary["ranking"].as_array().expect("ranking");
    assert_eq!(
        entries.len(),
        profiles_rendered,
        "expected one entry per speech profile"
    );
    assert_eq!(
        ranking.len(),
        profiles_rendered,
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
fn speech_pack_reports_intelligibility_vs_ugliness_tradeoffs() {
    let dir = temp_dir("speech_pack_tradeoff_contract");
    let out_dir = dir.join("speech_pack");

    run_ok(&[
        "speech-pack",
        "--text",
        "CLEAR WORDS SHOULD FIGHT THE BEAUTIFUL AWFUL MACHINE",
        "--out-dir",
        out_dir.to_str().expect("out dir"),
        "--sample-rate",
        "22050",
        "--rank-by",
        "balanced",
        "--top",
        "3",
    ]);

    let summary_path = out_dir.join("summary.json");
    let summary: Value =
        serde_json::from_str(&fs::read_to_string(&summary_path).expect("summary read"))
            .expect("summary json");

    let report = summary["tradeoff_report"]
        .as_object()
        .expect("summary should include a speech-pack tradeoff_report");
    for field in [
        "rank_by",
        "ugliness_metric",
        "intelligibility_metric",
        "ugliness_weight",
        "intelligibility_weight",
        "rank_formula",
        "summary",
    ] {
        assert!(
            report.contains_key(field),
            "tradeoff_report should include {field}: {report:#?}"
        );
    }
    let ugly_weight = report["ugliness_weight"].as_f64().expect("ugliness_weight");
    let intel_weight = report["intelligibility_weight"]
        .as_f64()
        .expect("intelligibility_weight");
    assert!(
        (ugly_weight + intel_weight - 1.0).abs() < 1e-6,
        "balanced tradeoff weights should sum to 1.0"
    );

    let ranking = summary["ranking"].as_array().expect("ranking");
    assert!(!ranking.is_empty(), "ranking should not be empty");
    for row in ranking {
        let tradeoff = row["tradeoff"]
            .as_object()
            .expect("ranking rows should explain their tradeoff");
        for field in [
            "colbys",
            "intelligibility_index",
            "rank_score",
            "ugliness_contribution",
            "intelligibility_contribution",
            "explanation",
        ] {
            assert!(
                tradeoff.contains_key(field),
                "ranking tradeoff should include {field}: {tradeoff:#?}"
            );
        }

        let colbys = tradeoff["colbys"].as_f64().expect("colbys");
        let intelligibility = tradeoff["intelligibility_index"]
            .as_f64()
            .expect("intelligibility_index");
        assert!(
            (-1000.0..=1000.0).contains(&colbys),
            "tradeoff colbys should stay bounded"
        );
        assert!(
            (0.0..=1000.0).contains(&intelligibility),
            "tradeoff intelligibility should be reported on a 0..1000 scale"
        );
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
