use super::*;

pub(super) fn render(args: RenderArgs) -> Result<()> {
    let randomness = RandomnessControls::from(&args.randomness);
    let output_encoding = args.output_format.output_encoding()?;
    let engine = engine_from_args(
        args.backend,
        args.jobs,
        args.gpu_drive,
        args.gpu_crush_bits,
        args.gpu_crush_mix,
    );
    let options = RenderOptions {
        duration: args.duration,
        sample_rate: args.sample_rate,
        seed: Some(apply_seed_controls(args.seed, randomness)),
        style: args.style.into(),
        gain: args.gain,
        normalize: !args.no_normalize,
        normalize_dbfs: args.normalize_dbfs,
        output_encoding,
    };
    let options = randomize_render_options(options, randomness);
    let summary = render_to_wav_with_engine(&args.output, &options, &engine)
        .with_context(|| format!("failed to write {}", args.output.display()))?;

    println!(
        "Rendered {} ({} frames @ {} Hz, style={}, seed={}, format={}, backend={} -> {}, jobs={})",
        summary.output.display(),
        summary.frames,
        summary.sample_rate,
        summary.style,
        summary.seed,
        summary.output_encoding,
        summary.backend_requested,
        summary.backend_active,
        summary.jobs
    );
    Ok(())
}

pub(super) fn piece(args: PieceArgs) -> Result<()> {
    let randomness = RandomnessControls::from(&args.randomness);
    let output_encoding = args.output_format.output_encoding()?;
    let base_seed = apply_seed_controls(args.seed, randomness);
    let layout = args
        .layout
        .as_deref()
        .map(parse_surround_layout)
        .transpose()?;
    let engine = engine_from_args(
        args.backend,
        args.jobs,
        args.gpu_drive,
        args.gpu_crush_bits,
        args.gpu_crush_mix,
    );
    let styles = selected_pack_styles(&args.styles);
    let options = PieceOptions {
        duration: randomize_duration(args.duration, randomness, base_seed),
        sample_rate: args.sample_rate,
        seed: Some(base_seed),
        styles,
        layout,
        channels: layout.map(|l| l.channels()).unwrap_or(args.channels),
        gain: randomize_gain(
            args.gain,
            randomness,
            derived_seed_label(base_seed, 0xA11C_E001),
        ),
        normalize: !args.no_normalize,
        normalize_dbfs: randomize_normalize_dbfs(
            args.normalize_dbfs,
            randomness,
            derived_seed_label(base_seed, 0xA11C_E002),
        ),
        output_encoding,
        events_per_second: (args.events_per_second
            * symmetric_factor(
                derived_seed_label(base_seed, 0xA11C_E003),
                randomness_amount(randomness.randomness, randomness.density_randomness),
                0.55,
            ))
        .clamp(0.1, 10_000.0),
        min_event_duration: (args.min_event_duration
            * symmetric_factor(
                derived_seed_label(base_seed, 0xA11C_E004),
                randomness_amount(randomness.randomness, randomness.timing_randomness),
                0.45,
            ))
        .clamp(0.005, args.duration.max(0.005)),
        max_event_duration: (args.max_event_duration
            * symmetric_factor(
                derived_seed_label(base_seed, 0xA11C_E005),
                randomness_amount(randomness.randomness, randomness.timing_randomness),
                0.45,
            ))
        .clamp(0.005, args.duration.max(0.005)),
        min_pan_width: (args.min_pan_width
            * symmetric_factor(
                derived_seed_label(base_seed, 0xA11C_E006),
                randomness_amount(randomness.randomness, randomness.spectral_randomness),
                0.5,
            ))
        .clamp(0.05, 64.0),
        max_pan_width: (args.max_pan_width
            * symmetric_factor(
                derived_seed_label(base_seed, 0xA11C_E007),
                randomness_amount(randomness.randomness, randomness.spectral_randomness),
                0.5,
            ))
        .clamp(0.05, 64.0),
    };
    let options = PieceOptions {
        min_event_duration: options.min_event_duration.min(options.duration),
        max_event_duration: options
            .max_event_duration
            .max(options.min_event_duration)
            .min(options.duration),
        max_pan_width: options.max_pan_width.max(options.min_pan_width),
        ..options
    };
    let summary = render_piece_to_wav_with_engine(&args.output, &options, &engine)
        .with_context(|| format!("failed to write {}", args.output.display()))?;

    println!(
        "Rendered piece {} ({} frames @ {} Hz, channels={}, layout={}, events={}, seed={}, format={}, backend={} -> {}, jobs={})",
        summary.output.display(),
        summary.frames,
        summary.sample_rate,
        summary.channels,
        summary.layout.as_deref().unwrap_or("custom"),
        summary.events,
        summary.seed,
        summary.output_encoding,
        summary.backend_requested,
        summary.backend_active,
        summary.jobs
    );
    Ok(())
}

pub(super) fn speech(args: SpeechArgs) -> Result<()> {
    let randomness = RandomnessControls::from(&args.randomness);
    let output_encoding = args.output_format.output_encoding()?;
    let jobs = if args.jobs == 0 {
        default_jobs()
    } else {
        args.jobs
    };
    let engine = engine_from_backend(
        args.backend.into(),
        jobs,
        args.gpu_drive,
        args.gpu_crush_bits,
        args.gpu_crush_mix,
    );
    let text = if let Some(path) = args.text_file.as_ref() {
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?
    } else {
        args.text
            .clone()
            .unwrap_or_else(|| "HELLO FROM UGLY SOUND GENERATOR".to_string())
    };
    let opts = SpeechRenderOptions {
        text,
        input_mode: args.input_mode.into(),
        sample_rate: args.sample_rate,
        seed: Some(apply_seed_controls(args.seed, randomness)),
        normalize_text: !args.no_normalize_text,
        chip_profile: args.profile.into(),
        primary_osc: args.primary_osc.into(),
        secondary_osc: args.secondary_osc.into(),
        tertiary_osc: args.tertiary_osc.into(),
        gain: args.gain,
        normalize: !args.no_normalize,
        normalize_dbfs: args.normalize_dbfs,
        output_encoding,
        units_per_second: args.units_per_second,
        pitch_hz: args.pitch_hz,
        pitch_jitter: args.pitch_jitter,
        vibrato_hz: args.vibrato_hz,
        vibrato_depth: args.vibrato_depth,
        duty_cycle: args.duty_cycle,
        formant_shift: args.formant_shift,
        consonant_noise: args.consonant_noise,
        vowel_mix: args.vowel_mix,
        hiss: args.hiss,
        buzz: args.buzz,
        fold: args.fold,
        chaos: args.chaos,
        robotize: args.robotize,
        glide: args.glide,
        monotone: args.monotone,
        emphasis: args.emphasis,
        word_accent: args.word_accent,
        sentence_lilt: args.sentence_lilt,
        paragraph_decline: args.paragraph_decline,
        word_gap_ms: args.word_gap_ms,
        sentence_gap_ms: args.sentence_gap_ms,
        paragraph_gap_ms: args.paragraph_gap_ms,
        punctuation_gap_ms: args.punctuation_gap_ms,
        attack_ms: args.attack_ms,
        release_ms: args.release_ms,
        bitcrush_bits: args.bitcrush_bits,
        sample_hold_hz: args.sample_hold_hz,
        ring_mix: args.ring_mix,
        sub_mix: args.sub_mix,
        nasal: args.nasal,
        throat: args.throat,
        drift: args.drift,
        resampler_grit: args.resampler_grit,
    };
    let opts = randomize_speech_options(opts, randomness);
    let artifacts = render_speech_with_artifacts_to_wav_with_engine(&args.output, &opts, &engine)?;
    let summary = artifacts.summary.clone();
    println!(
        "Rendered speech to {} ({} frames @ {} Hz, profile={}, backend={}, mode={}, units={}, phonemes={}, format={})",
        args.output.display(),
        summary.frames,
        summary.sample_rate,
        summary.chip_profile,
        summary.backend_kind,
        summary.input_mode,
        summary.units_rendered,
        summary.phonemes_rendered,
        summary.output_encoding
    );
    println!(
        "oscillators: primary={}, secondary={}, tertiary={}",
        opts.primary_osc, opts.secondary_osc, opts.tertiary_osc
    );
    println!("seed: {}", summary.seed);
    if let Some(path) = args.analysis_json.as_ref() {
        let analysis = analyze_wav_with_options(
            &args.output,
            &AnalyzeOptions {
                model: AnalyzeModel::Psycho,
                fft_size: 2048,
                hop_size: 512,
                joke: false,
            },
        )?;
        let payload = serde_json::json!({
            "summary": &summary,
            "analysis": analysis,
            "intelligibility": score_speech_intelligibility(&summary, &artifacts.timeline),
        });
        write_json(path, &payload)
            .with_context(|| format!("failed to write {}", path.display()))?;
        println!("analysis json: {}", path.display());
    }
    if let Some(path) = args.timeline_json.as_ref() {
        write_json(path, &artifacts.timeline)
            .with_context(|| format!("failed to write {}", path.display()))?;
        println!("timeline json: {}", path.display());
    }

    if args.play {
        let played = try_play_audio(&args.output);
        if !played {
            println!(
                "Could not auto-play audio. Open manually: {}",
                args.output.display()
            );
        }
    }
    Ok(())
}

pub(super) fn analyze(args: AnalyzeArgs) -> Result<()> {
    if args.timeline {
        return analyze_timeline(args);
    }

    let options = AnalyzeOptions {
        model: args.model.into(),
        fft_size: args.fft_size,
        hop_size: args.hop_size,
        joke: args.joke,
    };
    let report = analyze_wav_with_options(&args.input, &options)
        .with_context(|| format!("failed to analyze {}", args.input.display()))?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("Analysis for {}", args.input.display());
    println!("model: {}", report.model);
    println!(
        "score_profile: {} v{} ({} {}..{}; neutral {})",
        report.score_metadata.profile,
        report.score_metadata.version,
        report.score_metadata.unit,
        report.score_metadata.min,
        report.score_metadata.max,
        report.score_metadata.neutral
    );
    println!(
        "score_calibrated_from_listening_tests: {}",
        report.score_metadata.calibrated_from_listening_tests
    );
    println!("sample_rate_hz: {}", report.basic.sample_rate);
    println!("channels: {}", report.basic.channels);
    println!("duration_s: {:.3}", report.basic.duration_s);
    println!("peak_dbfs: {:.2}", report.basic.peak_dbfs);
    println!("rms_dbfs: {:.2}", report.basic.rms_dbfs);
    println!("crest_factor_db: {:.2}", report.basic.crest_factor_db);
    println!("zero_crossing_rate: {:.4}", report.basic.zero_crossing_rate);
    println!("clipped_pct: {:.2}%", report.basic.clipped_pct);
    println!("harshness_ratio: {:.3}", report.basic.harshness_ratio);
    println!("basic.colbys: {:.0} Co", report.basic.colbys);

    if let Some(psycho) = &report.psycho {
        println!("psycho.clip_norm: {:.3}", psycho.clip_norm);
        println!("psycho.harshness_norm: {:.3}", psycho.harshness_norm);
        println!("psycho.roughness_norm: {:.3}", psycho.roughness_norm);
        println!("psycho.sharpness_norm: {:.3}", psycho.sharpness_norm);
        println!("psycho.dissonance_norm: {:.3}", psycho.dissonance_norm);
        println!("psycho.transient_norm: {:.3}", psycho.transient_norm);
        println!("psycho.harmonicity_norm: {:.3}", psycho.harmonicity_norm);
        println!(
            "psycho.inharmonicity_norm: {:.3}",
            psycho.inharmonicity_norm
        );
        println!(
            "psycho.binaural_beat_norm: {:.3}",
            psycho.binaural_beat_norm
        );
        println!(
            "psycho.beat_conflict_norm: {:.3}",
            psycho.beat_conflict_norm
        );
        println!(
            "psycho.tritone_tension_norm: {:.3}",
            psycho.tritone_tension_norm
        );
        println!("psycho.wolf_fifth_norm: {:.3}", psycho.wolf_fifth_norm);
        println!("psycho.weighted_sum: {:.3}", psycho.weighted_sum);
        println!("psycho_fft_size: {}", psycho.fft_size);
        println!("psycho_hop_size: {}", psycho.hop_size);
    }

    if let Some(joke) = &report.joke {
        println!("joke.uglierbasis_index: {:.0} Co", joke.uglierbasis_index);
        println!("joke.verdict: {}", joke.verdict);
        println!(
            "joke.academic_cluster_norm: {:.3}",
            joke.academic_cluster_norm
        );
        println!(
            "joke.bureaucratic_overhead_norm: {:.3}",
            joke.bureaucratic_overhead_norm
        );
        println!("joke.all_high_bonus_norm: {:.3}", joke.all_high_bonus_norm);
        println!(
            "joke.harmonicity_relief_norm: {:.3}",
            joke.harmonicity_relief_norm
        );
        println!("joke.weighted_sum: {:.3}", joke.weighted_sum);
    }

    println!("colbys: {:.0} Co  (Colbys)", report.colbys);
    Ok(())
}

fn analyze_timeline(args: AnalyzeArgs) -> Result<()> {
    let opts = TimelineOptions {
        window_ms: args.timeline_window_ms,
        hop_ms: args.timeline_hop_ms,
    };
    let frames = analyze_wav_timeline(&args.input, &opts)
        .with_context(|| format!("failed to compute timeline for {}", args.input.display()))?;

    let emit = |text: String| -> Result<()> {
        if let Some(path) = &args.timeline_output {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)?;
                }
            }
            fs::write(path, text)?;
        } else {
            print!("{}", text);
        }
        Ok(())
    };

    match args.timeline_format {
        TimelineFormatArg::Json => emit(serde_json::to_string_pretty(&frames)? + "\n"),
        TimelineFormatArg::Csv => {
            let mut out =
                String::from("time_s,colbys,clipped_pct,harshness_ratio,zero_crossing_rate\n");
            for f in &frames {
                out.push_str(&format!(
                    "{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                    f.time_s, f.colbys, f.clipped_pct, f.harshness_ratio, f.zero_crossing_rate
                ));
            }
            emit(out)
        }
    }
}

pub(super) fn speech_pack(args: SpeechPackArgs) -> Result<()> {
    let randomness = RandomnessControls::from(&args.randomness);
    let output_encoding = args.output_format.output_encoding()?;
    let text = if let Some(path) = args.text_file {
        fs::read_to_string(&path)
            .with_context(|| format!("failed to read text file {}", path.display()))?
    } else {
        args.text
            .unwrap_or_else(|| "UGLY SOUND GENERATOR".to_string())
    };
    let analyze_options = AnalyzeOptions {
        model: args.model.into(),
        fft_size: args.fft_size,
        hop_size: args.hop_size,
        joke: false,
    };
    let engine = engine_from_args(
        args.backend,
        args.jobs,
        args.gpu_drive,
        args.gpu_crush_bits,
        args.gpu_crush_mix,
    );
    let plan = resolve_backend_plan(&engine)?;
    fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("failed to create {}", args.out_dir.display()))?;

    let base_seed = apply_seed_controls(args.seed, randomness);
    let profiles = SpeechChipProfile::ALL;
    let tasks: Vec<(usize, SpeechChipProfile)> = profiles.iter().copied().enumerate().collect();
    let rank_by = match args.rank_by {
        SpeechPackRankArg::Ugliness => "ugliness",
        SpeechPackRankArg::Intelligibility => "intelligibility",
        SpeechPackRankArg::Balanced => "balanced",
    };

    let pool = ThreadPoolBuilder::new()
        .num_threads(plan.jobs)
        .build()
        .context("failed to build speech-pack worker pool")?;

    let rendered: Vec<Result<(usize, SpeechPackEntry)>> = pool.install(|| {
        tasks
            .par_iter()
            .map(|(idx, profile)| {
                let seed = derived_batch_seed(base_seed, *idx as u64, args.seed_stride);
                let output = args
                    .out_dir
                    .join(format!("{:02}_{}.wav", *idx + 1, profile.as_str()));
                let opts = SpeechRenderOptions {
                    text: text.clone(),
                    input_mode: args.input_mode.into(),
                    sample_rate: args.sample_rate,
                    seed: Some(seed),
                    normalize_text: true,
                    chip_profile: *profile,
                    pitch_hz: args.pitch_hz,
                    normalize: !args.no_normalize,
                    normalize_dbfs: args.normalize_dbfs,
                    output_encoding,
                    ..SpeechRenderOptions::default()
                };
                let opts = randomize_speech_options(opts, randomness);
                let artifacts =
                    render_speech_with_artifacts_to_wav_with_engine(&output, &opts, &engine)
                        .with_context(|| format!("failed to render {}", output.display()))?;
                let analysis = analyze_wav_with_options(&output, &analyze_options)
                    .with_context(|| format!("failed to analyze {}", output.display()))?;
                let intelligibility =
                    score_speech_intelligibility(&artifacts.summary, &artifacts.timeline);
                Ok((
                    *idx,
                    SpeechPackEntry {
                        profile: profile.as_str().to_string(),
                        seed,
                        output: output.display().to_string(),
                        colbys: analysis.colbys,
                        intelligibility,
                        analysis,
                    },
                ))
            })
            .collect()
    });

    let mut ordered: Vec<(usize, SpeechPackEntry)> = Vec::with_capacity(rendered.len());
    for entry in rendered {
        ordered.push(entry?);
    }
    ordered.sort_by_key(|(idx, _)| *idx);
    let entries: Vec<SpeechPackEntry> = ordered.into_iter().map(|(_, e)| e).collect();

    let mut ranked = entries.clone();
    ranked.sort_by(|a, b| {
        speech_pack_rank_score(b, args.rank_by).total_cmp(&speech_pack_rank_score(a, args.rank_by))
    });
    let ranking: Vec<SpeechPackRankingEntry> = ranked
        .iter()
        .enumerate()
        .map(|(i, e)| SpeechPackRankingEntry {
            rank: i + 1,
            profile: e.profile.clone(),
            output: e.output.clone(),
            colbys: e.colbys,
            intelligibility_index: e.intelligibility.intelligibility_index,
            rank_score: speech_pack_rank_score(e, args.rank_by),
            basic_colbys: e.analysis.basic.colbys,
            seed: e.seed,
        })
        .collect();

    let summary = SpeechPackSummary {
        generated_unix_s: now_unix_s(),
        model: analyze_options.model.as_str().to_string(),
        rank_by: rank_by.to_string(),
        text: text.chars().take(80).collect(),
        sample_rate_hz: args.sample_rate,
        backend_requested: plan.requested.as_str().to_string(),
        backend_active: plan.active.as_str().to_string(),
        jobs: plan.jobs,
        base_seed,
        profiles_rendered: entries.len(),
        entries,
        ranking,
    };

    let summary_path = args
        .summary
        .unwrap_or_else(|| args.out_dir.join("summary.json"));
    let csv_path = args.csv.unwrap_or_else(|| args.out_dir.join("ranking.csv"));
    let html_path = args
        .html
        .unwrap_or_else(|| args.out_dir.join("report.html"));
    write_json(&summary_path, &summary)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    write_speech_pack_csv(&csv_path, &summary)
        .with_context(|| format!("failed to write {}", csv_path.display()))?;
    write_speech_pack_html(&html_path, &summary)
        .with_context(|| format!("failed to write {}", html_path.display()))?;

    println!(
        "Rendered speech pack: {} profiles -> {} (backend={} -> {}, jobs={})",
        summary.profiles_rendered,
        args.out_dir.display(),
        summary.backend_requested,
        summary.backend_active,
        summary.jobs
    );
    println!("Text: \"{}\"", summary.text);
    println!("Ranking: {}", summary.rank_by);
    println!("Summary: {}", summary_path.display());
    println!("CSV: {}", csv_path.display());
    println!("HTML: {}", html_path.display());
    println!("Top ugliest:");
    for row in summary.ranking.iter().take(args.top.max(1)) {
        println!(
            "  {:>2}. {:<14} ugly {:>5.1}  intel {:>5.1}  score {:>5.1}  {}",
            row.rank,
            row.profile,
            row.colbys,
            row.intelligibility_index,
            row.rank_score,
            row.output
        );
    }
    Ok(())
}

pub(super) fn render_pack(args: RenderPackArgs) -> Result<()> {
    let randomness = RandomnessControls::from(&args.randomness);
    let output_encoding = args.output_format.output_encoding()?;
    let analyze_options = AnalyzeOptions {
        model: args.model.into(),
        fft_size: args.fft_size,
        hop_size: args.hop_size,
        joke: false,
    };
    let engine = engine_from_args(
        args.backend,
        args.jobs,
        args.gpu_drive,
        args.gpu_crush_bits,
        args.gpu_crush_mix,
    );
    let plan = resolve_backend_plan(&engine)?;
    fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("failed to create {}", args.out_dir.display()))?;

    let base_seed = apply_seed_controls(args.seed, randomness);
    let styles_to_render = selected_pack_styles(&args.styles);
    let tasks: Vec<(usize, Style)> = styles_to_render.iter().copied().enumerate().collect();
    let pool = ThreadPoolBuilder::new()
        .num_threads(plan.jobs)
        .build()
        .context("failed to build pack worker pool")?;
    let rendered_entries: Vec<Result<(usize, PackEntry)>> = pool.install(|| {
        tasks
            .par_iter()
            .map(|(idx, style)| {
                let seed = derived_batch_seed(base_seed, *idx as u64, args.seed_stride);
                let output = args
                    .out_dir
                    .join(format!("{:02}_{}.wav", *idx + 1, style.as_str()));
                let render_options = RenderOptions {
                    duration: args.duration,
                    sample_rate: args.sample_rate,
                    seed: Some(seed),
                    style: *style,
                    gain: args.gain,
                    normalize: !args.no_normalize,
                    normalize_dbfs: args.normalize_dbfs,
                    output_encoding,
                };
                let render_options = randomize_render_options(render_options, randomness);
                render_to_wav_with_engine(&output, &render_options, &engine)
                    .with_context(|| format!("failed to render {}", output.display()))?;
                let analysis = analyze_wav_with_options(&output, &analyze_options)
                    .with_context(|| format!("failed to analyze {}", output.display()))?;
                Ok((
                    *idx,
                    PackEntry {
                        style: style.as_str().to_string(),
                        seed,
                        output: output.display().to_string(),
                        colbys: analysis.colbys,
                        analysis,
                    },
                ))
            })
            .collect()
    });
    let mut ordered_entries: Vec<(usize, PackEntry)> = Vec::with_capacity(rendered_entries.len());
    for entry in rendered_entries {
        ordered_entries.push(entry?);
    }
    ordered_entries.sort_by_key(|(idx, _)| *idx);
    let entries: Vec<PackEntry> = ordered_entries
        .into_iter()
        .map(|(_, entry)| entry)
        .collect();

    let mut ranked = entries.clone();
    ranked.sort_by(|a, b| b.colbys.total_cmp(&a.colbys));
    let ranking: Vec<PackRankingEntry> = ranked
        .iter()
        .enumerate()
        .map(|(i, e)| PackRankingEntry {
            rank: i + 1,
            style: e.style.clone(),
            output: e.output.clone(),
            colbys: e.colbys,
            seed: e.seed,
            basic_colbys: e.analysis.basic.colbys,
        })
        .collect();

    let summary = PackSummary {
        generated_unix_s: now_unix_s(),
        model: analyze_options.model.as_str().to_string(),
        duration_s: args.duration,
        sample_rate_hz: args.sample_rate,
        gain: args.gain,
        output_encoding: output_encoding.as_str().to_string(),
        backend_requested: plan.requested.as_str().to_string(),
        backend_active: plan.active.as_str().to_string(),
        jobs: plan.jobs,
        base_seed,
        styles_requested: styles_to_render
            .iter()
            .map(|s| s.as_str().to_string())
            .collect(),
        styles_rendered: entries.len(),
        entries,
        ranking,
    };

    let summary_path = args
        .summary
        .unwrap_or_else(|| args.out_dir.join("summary.json"));
    let csv_path = args.csv.unwrap_or_else(|| args.out_dir.join("ranking.csv"));
    let html_path = args
        .html
        .unwrap_or_else(|| args.out_dir.join("report.html"));
    write_json(&summary_path, &summary)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    write_pack_csv(&csv_path, &summary)
        .with_context(|| format!("failed to write {}", csv_path.display()))?;
    write_pack_html(&html_path, &summary)
        .with_context(|| format!("failed to write {}", html_path.display()))?;

    println!(
        "Rendered style pack: {} styles -> {} (backend={} -> {}, jobs={})",
        summary.styles_rendered,
        args.out_dir.display(),
        summary.backend_requested,
        summary.backend_active,
        summary.jobs
    );
    println!("Output encoding: {}", summary.output_encoding);
    println!("Summary: {}", summary_path.display());
    println!("CSV: {}", csv_path.display());
    println!("HTML: {}", html_path.display());
    println!("Top ugliest:");
    for row in summary.ranking.iter().take(args.top.max(1)) {
        println!(
            "  {:>2}. {:<8} {:>6.0} Co (basic {:>5.1})  {}",
            row.rank, row.style, row.colbys, row.basic_colbys, row.output
        );
    }
    Ok(())
}

pub(super) fn go(args: GoArgs) -> Result<()> {
    let randomness = RandomnessControls::from(&args.randomness);
    let output_encoding = args.output_format.output_encoding()?;
    let engine = engine_from_args(
        args.backend,
        args.jobs,
        args.gpu_drive,
        args.gpu_crush_bits,
        args.gpu_crush_mix,
    );
    let contour = parse_ugliness_contour(&args)?;
    let output = args
        .output
        .unwrap_or_else(|| default_go_output_path(&args.input));
    let flavor = args.flavor.map(Into::into);
    let seed = apply_seed_controls(args.seed, randomness);
    let target_colbys = randomize_go_colbys(args.level, randomness, seed);
    let normalize_dbfs = randomize_normalize_dbfs(args.normalize_dbfs, randomness, seed);

    let summary = if let Some(layout_text) = args.upmix.as_ref() {
        let layout = parse_surround_layout(layout_text)?;
        let coord_system: CoordSystem = args.coords.into();
        let locus_vals = parse_triplet_csv(&args.locus)
            .with_context(|| format!("invalid --locus '{}'", args.locus))?;
        let locus_xyz = point_to_xyz(coord_system, locus_vals[0], locus_vals[1], locus_vals[2]);
        let trajectory = parse_trajectory(&args.trajectory, coord_system)
            .with_context(|| format!("invalid --trajectory '{}'", args.trajectory))?;
        let spatial = SpatialGoOptions {
            layout,
            locus_xyz,
            trajectory,
        };
        go_ugly_upmix_file_with_engine_contour_encoding(
            &args.input,
            &output,
            target_colbys,
            flavor,
            Some(seed),
            !args.no_normalize,
            normalize_dbfs,
            spatial,
            contour.as_ref(),
            Some(args.sample_rate),
            output_encoding,
            &engine,
        )
    } else {
        go_ugly_file_with_engine_contour_encoding(
            &args.input,
            &output,
            target_colbys,
            flavor,
            Some(seed),
            !args.no_normalize,
            normalize_dbfs,
            contour.as_ref(),
            Some(args.sample_rate),
            output_encoding,
            &engine,
        )
    }
    .with_context(|| {
        format!(
            "failed to uglify {} -> {}",
            args.input.display(),
            output.display()
        )
    })?;

    println!(
        "GO complete: {} ({} frames @ {} Hz, channels={}, target_colbys={}, intensity={:.3}, type={}, layout={}, seed={}, format={}, backend={} -> {}, jobs={})",
        summary.output.display(),
        summary.frames,
        summary.sample_rate,
        summary.channels,
        summary.target_colbys,
        summary.target_intensity,
        summary.flavor,
        summary.layout.as_deref().unwrap_or("source"),
        summary.seed,
        summary.output_encoding,
        summary.backend_requested,
        summary.backend_active,
        summary.jobs
    );
    if contour.is_some() {
        println!("Applied contour: time-varying ugliness envelope from JSON");
    }
    Ok(())
}

pub(super) fn chain(args: ChainArgs) -> Result<()> {
    let randomness = RandomnessControls::from(&args.randomness);
    let output_encoding = args.output_format.output_encoding()?;
    let engine = engine_from_args(
        args.backend,
        args.jobs,
        args.gpu_drive,
        args.gpu_crush_bits,
        args.gpu_crush_mix,
    );
    let base_seed = apply_seed_controls(args.seed, randomness);
    let mut stage_tokens = Vec::new();
    if let Some(preset_name) = args.preset.as_ref() {
        let preset = load_chain_preset(preset_name)?;
        stage_tokens.extend(preset.stages);
    }
    stage_tokens.extend(args.stages.clone());

    let mut parsed = Vec::new();
    let mut invalid = Vec::new();
    for token in &stage_tokens {
        if let Some(stage) = parse_chain_stage(token) {
            parsed.push(stage);
        } else {
            invalid.push(token.clone());
        }
    }
    if !invalid.is_empty() {
        return Err(anyhow::anyhow!(
            "unknown chain stage(s): {}. valid styles: {}. valid effects: {}. use style:<name> or effect:<name> for explicit resolution.",
            invalid.join(", "),
            styles_csv(),
            effects_csv(),
        ));
    }

    let frames = render_chain_to_wav_with_engine(
        &args.output,
        &parsed,
        randomize_duration(args.duration, randomness, base_seed),
        args.sample_rate,
        output_encoding,
        randomize_gain(args.gain, randomness, derived_seed_label(base_seed, 0xC1A1)),
        !args.no_normalize,
        randomize_normalize_dbfs(
            args.normalize_dbfs,
            randomness,
            derived_seed_label(base_seed, 0xC1A2),
        ),
        base_seed,
        &engine,
    )?;

    println!(
        "Rendered chain to {} ({} frames @ {} Hz, format={}, stages: {})",
        args.output.display(),
        frames,
        args.sample_rate,
        output_encoding,
        format_chain(&parsed)
    );
    println!("seed: {}", base_seed);

    if args.play {
        let played = try_play_audio(&args.output);
        if !played {
            println!(
                "Could not auto-play audio. Open manually: {}",
                args.output.display()
            );
        }
    }
    Ok(())
}

pub(super) fn styles() -> Result<()> {
    for style in available_styles() {
        println!("{style}");
    }
    Ok(())
}

pub(super) fn backends() -> Result<()> {
    let caps = backend_capabilities();
    let status = backend_status_report();
    let effective = engine_from_backend(RenderBackend::Auto, default_jobs(), None, None, None);
    let cuda_disabled = std::env::var_os("USG_DISABLE_CUDA").is_some();

    println!(
        "CPU:   {}",
        if status.cpu.available {
            "available"
        } else {
            "not available"
        }
    );
    println!(
        "Metal: {}",
        if caps.metal {
            "available (build with --features metal)"
        } else {
            "not available"
        }
    );
    println!(
        "CUDA:  {}",
        if caps.cuda {
            "available (build with --features cuda)"
        } else {
            "not available"
        }
    );
    println!("Default jobs: {}", default_jobs());
    println!(
        "GPU post-FX defaults: drive={:.2}, crush_bits={:.2}, crush_mix={:.2}",
        effective.gpu_drive, effective.gpu_crush_bits, effective.gpu_crush_mix
    );
    println!("GPU post-FX env vars: USG_GPU_DRIVE, USG_GPU_CRUSH_BITS, USG_GPU_CRUSH_MIX");
    println!("CPU detail: {}", status.cpu.detail);
    println!(
        "Metal detail: feature_built={}, target_os_macos={}, runtime_available={}",
        cfg!(feature = "metal"),
        cfg!(target_os = "macos"),
        caps.metal
    );
    println!("Metal probe: {}", status.metal.detail);
    println!(
        "CUDA detail: feature_built={}, disabled_by_env={}, runtime_available={}",
        cfg!(feature = "cuda"),
        cuda_disabled,
        caps.cuda
    );
    println!("CUDA probe: {}", status.cuda.detail);
    if !cfg!(feature = "metal") {
        println!("Metal note: rebuild with --features metal to enable Metal acceleration.");
    } else if !cfg!(target_os = "macos") {
        println!("Metal note: Metal rendering is only supported on macOS targets.");
    } else if !caps.metal {
        println!("Metal note: no usable Metal device/runtime was detected.");
    }
    if cuda_disabled {
        println!("CUDA note: USG_DISABLE_CUDA is set, so CUDA was intentionally disabled.");
    } else if !cfg!(feature = "cuda") {
        println!("CUDA note: rebuild with --features cuda to enable CUDA acceleration.");
    } else if !caps.cuda {
        println!("CUDA note: a CUDA-capable runtime/device was not detected at startup.");
    }
    Ok(())
}

pub(super) fn presets(args: PresetsArgs) -> Result<()> {
    let entries = builtin_preset_entries(args.kind)?;

    if let Some(name) = args.show.as_ref() {
        let path = resolve_preset_path(name, &entries)?;
        let info = entries
            .iter()
            .find(|entry| entry.path == path.display().to_string())
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("internal error: resolved preset missing from index"))?;
        let text = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if args.json {
            if info.kind == "contour" {
                let contour: UglinessContour = serde_json::from_str(&text)
                    .with_context(|| format!("invalid contour JSON in {}", path.display()))?;
                println!("{}", serde_json::to_string_pretty(&contour)?);
            } else {
                let preset: ChainPreset = serde_json::from_str(&text)
                    .with_context(|| format!("invalid chain preset JSON in {}", path.display()))?;
                println!("{}", serde_json::to_string_pretty(&preset)?);
            }
            return Ok(());
        }
        println!("Preset: {}", info.name);
        println!("Kind: {}", info.kind);
        println!("Version: {}", info.version);
        println!("Path: {}", path.display());
        println!("Summary: {}", info.summary);
        println!();
        println!("{}", text);
        return Ok(());
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&entries)?);
        return Ok(());
    }

    println!("Built-in presets: {}", entries.len());
    for entry in &entries {
        if args.paths {
            println!(
                "  {:<8} v{:<2} {:<28} {:<44} {}",
                entry.kind, entry.version, entry.name, entry.summary, entry.path
            );
        } else {
            println!(
                "  {:<8} v{:<2} {:<28} {}",
                entry.kind, entry.version, entry.name, entry.summary
            );
        }
    }
    println!("Use 'usg presets --show <name>' to inspect one preset.");
    Ok(())
}

pub(super) fn benchmark(args: BenchmarkArgs) -> Result<()> {
    let randomness = RandomnessControls::from(&args.randomness);
    let output_encoding = args.output_format.output_encoding()?;
    let runs = args.runs.max(1);
    let jobs = if args.jobs == 0 {
        default_jobs()
    } else {
        args.jobs
    };
    let base_seed = apply_seed_controls(args.seed, randomness);
    let caps = backend_capabilities();

    let mut backends = vec![RenderBackend::Cpu];
    if caps.metal {
        backends.push(RenderBackend::Metal);
    }
    if caps.cuda {
        backends.push(RenderBackend::Cuda);
    }

    println!(
        "Benchmarking {} backend(s), runs={}, duration={:.2}s, sample_rate={} Hz, jobs={}, format={}",
        backends.len(),
        runs,
        args.duration,
        args.sample_rate,
        jobs,
        output_encoding
    );

    let mut rows: Vec<BenchmarkEntry> = Vec::new();
    for backend in backends {
        let engine = engine_from_backend(
            backend,
            jobs,
            args.gpu_drive,
            args.gpu_crush_bits,
            args.gpu_crush_mix,
        );
        let mut run_ms = Vec::with_capacity(runs);
        for run_idx in 0..runs {
            let out_path = std::env::temp_dir().join(format!(
                "usg_bench_{}_{}_{}.wav",
                backend.as_str(),
                std::process::id(),
                run_idx
            ));
            let opts = RenderOptions {
                duration: randomize_duration(
                    args.duration,
                    randomness,
                    derived_batch_seed(base_seed, run_idx as u64, args.seed_stride),
                ),
                sample_rate: args.sample_rate,
                seed: Some(derived_batch_seed(
                    base_seed,
                    run_idx as u64,
                    args.seed_stride,
                )),
                style: args.style.into(),
                gain: randomize_gain(
                    args.gain,
                    randomness,
                    derived_seed_label(base_seed, run_idx as u64 ^ 0xB00B),
                ),
                normalize: !args.no_normalize,
                normalize_dbfs: randomize_normalize_dbfs(
                    args.normalize_dbfs,
                    randomness,
                    derived_seed_label(base_seed, run_idx as u64 ^ 0xB00C),
                ),
                output_encoding,
            };
            let t0 = Instant::now();
            render_to_wav_with_engine(&out_path, &opts, &engine)
                .with_context(|| format!("benchmark render failed for {}", backend))?;
            run_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);
            let _ = fs::remove_file(out_path);
        }
        let average_ms = run_ms.iter().sum::<f64>() / run_ms.len() as f64;
        let min_ms = run_ms
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap_or(average_ms);
        let max_ms = run_ms
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap_or(average_ms);
        rows.push(BenchmarkEntry {
            rank: 0,
            backend: backend.as_str().to_string(),
            average_ms,
            min_ms,
            max_ms,
            realtime_factor: if average_ms > 0.0 {
                args.duration * 1_000.0 / average_ms
            } else {
                f64::INFINITY
            },
            runs,
            run_ms,
        });
    }

    rows.sort_by(|a, b| a.average_ms.total_cmp(&b.average_ms));
    for (idx, row) in rows.iter_mut().enumerate() {
        row.rank = idx + 1;
    }

    let report = BenchmarkReport {
        generated_unix_s: now_unix_s(),
        runs,
        duration_s: args.duration,
        sample_rate_hz: args.sample_rate,
        style: Style::from(args.style).as_str().to_string(),
        gain: args.gain,
        normalize: !args.no_normalize,
        normalize_dbfs: args.normalize_dbfs,
        output_encoding: output_encoding.as_str().to_string(),
        jobs,
        gpu_drive: engine_from_backend(
            RenderBackend::Auto,
            jobs,
            args.gpu_drive,
            args.gpu_crush_bits,
            args.gpu_crush_mix,
        )
        .gpu_drive,
        gpu_crush_bits: engine_from_backend(
            RenderBackend::Auto,
            jobs,
            args.gpu_drive,
            args.gpu_crush_bits,
            args.gpu_crush_mix,
        )
        .gpu_crush_bits,
        gpu_crush_mix: engine_from_backend(
            RenderBackend::Auto,
            jobs,
            args.gpu_drive,
            args.gpu_crush_bits,
            args.gpu_crush_mix,
        )
        .gpu_crush_mix,
        rows,
    };

    println!("Average render time:");
    for row in &report.rows {
        println!(
            "  {:>2}. {:<5} {:>8.3} ms  min {:>8.3}  max {:>8.3}  x{:>6.2} realtime",
            row.rank, row.backend, row.average_ms, row.min_ms, row.max_ms, row.realtime_factor
        );
    }

    if let Some(path) = args.json_output.as_ref() {
        write_json(path, &report).with_context(|| format!("failed to write {}", path.display()))?;
        println!("JSON: {}", path.display());
    }
    if let Some(path) = args.csv_output.as_ref() {
        write_benchmark_csv(path, &report)
            .with_context(|| format!("failed to write {}", path.display()))?;
        println!("CSV: {}", path.display());
    }
    Ok(())
}

pub(super) fn marathon(args: MarathonArgs) -> Result<()> {
    let randomness = RandomnessControls::from(&args.randomness);
    let output_encoding = args.output_format.output_encoding()?;
    if args.count == 0 {
        return Err(anyhow::anyhow!("count must be >= 1"));
    }
    if args.min_duration <= 0.0 || args.max_duration <= 0.0 {
        return Err(anyhow::anyhow!("durations must be > 0"));
    }
    if args.min_duration > args.max_duration {
        return Err(anyhow::anyhow!("min-duration must be <= max-duration"));
    }

    fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("failed to create {}", args.out_dir.display()))?;
    let styles = selected_pack_styles(&args.styles);
    if styles.is_empty() {
        return Err(anyhow::anyhow!("no styles selected"));
    }

    let base_seed = apply_seed_controls(args.seed, randomness);
    let engine = engine_from_args(
        args.backend,
        args.jobs,
        args.gpu_drive,
        args.gpu_crush_bits,
        args.gpu_crush_mix,
    );
    let plan = resolve_backend_plan(&engine)?;
    let tasks: Vec<usize> = (0..args.count).collect();
    let pool = ThreadPoolBuilder::new()
        .num_threads(plan.jobs)
        .build()
        .context("failed to build marathon worker pool")?;

    let rendered: Vec<Result<(usize, MarathonEntry)>> = pool.install(|| {
        tasks
            .par_iter()
            .map(|idx| {
                let slot_seed = derived_batch_seed(base_seed, *idx as u64, args.seed_stride);
                let style = styles[*idx % styles.len()];
                let duration = duration_for_slot(slot_seed, args.min_duration, args.max_duration);
                let output = args.out_dir.join(format!(
                    "{:05}_{}_s{slot_seed}.wav",
                    idx + 1,
                    style.as_str()
                ));
                let opts = RenderOptions {
                    duration: randomize_duration(duration, randomness, slot_seed),
                    sample_rate: args.sample_rate,
                    seed: Some(slot_seed),
                    style,
                    gain: randomize_gain(
                        args.gain,
                        randomness,
                        derived_seed_label(slot_seed, 0xA11C),
                    ),
                    normalize: !args.no_normalize,
                    normalize_dbfs: randomize_normalize_dbfs(
                        args.normalize_dbfs,
                        randomness,
                        derived_seed_label(slot_seed, 0xA11D),
                    ),
                    output_encoding,
                };
                render_to_wav_with_engine(&output, &opts, &engine)
                    .with_context(|| format!("failed to render {}", output.display()))?;
                Ok((
                    *idx,
                    MarathonEntry {
                        index: *idx + 1,
                        style: style.as_str().to_string(),
                        duration_s: duration,
                        seed: slot_seed,
                        output: output.display().to_string(),
                    },
                ))
            })
            .collect()
    });

    let mut ordered: Vec<(usize, MarathonEntry)> = Vec::with_capacity(rendered.len());
    for row in rendered {
        ordered.push(row?);
    }
    ordered.sort_by_key(|(idx, _)| *idx);
    let entries: Vec<MarathonEntry> = ordered.into_iter().map(|(_, entry)| entry).collect();

    let manifest = MarathonManifest {
        generated_unix_s: now_unix_s(),
        count: entries.len(),
        sample_rate_hz: args.sample_rate,
        gain: args.gain,
        output_encoding: output_encoding.as_str().to_string(),
        normalize: !args.no_normalize,
        normalize_dbfs: args.normalize_dbfs,
        backend_requested: plan.requested.as_str().to_string(),
        backend_active: plan.active.as_str().to_string(),
        jobs: plan.jobs,
        base_seed,
        min_duration_s: args.min_duration,
        max_duration_s: args.max_duration,
        styles: styles.iter().map(|s| s.as_str().to_string()).collect(),
        entries,
    };

    let manifest_path = args
        .manifest
        .unwrap_or_else(|| args.out_dir.join("manifest.json"));
    write_json(&manifest_path, &manifest)
        .with_context(|| format!("failed to write {}", manifest_path.display()))?;

    println!(
        "Marathon complete: {} files in {} (backend={} -> {}, jobs={})",
        manifest.count,
        args.out_dir.display(),
        manifest.backend_requested,
        manifest.backend_active,
        manifest.jobs
    );
    println!("Output encoding: {}", manifest.output_encoding);
    println!("Manifest: {}", manifest_path.display());
    println!("Examples:");
    for row in manifest.entries.iter().take(5) {
        println!(
            "  {:>5}. {:<8} {:>7.2}s  {}",
            row.index, row.style, row.duration_s, row.output
        );
    }
    Ok(())
}
