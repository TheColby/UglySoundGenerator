# Psychoacoustics, Equations, And References

This document explains the psychoacoustic side of USG without pretending the current model is more authoritative than it is.

## Model Position

USG's psycho profile is a **transparent heuristic** inspired by psychoacoustic literature, interval tension, and roughness work.

It is useful for ranking and steering ugly sounds.
It is not a standardized clinical or perceptual metric.

## Feature Families

The current psycho profile looks at the following normalized components:

- clipping pressure
- roughness
- sharpness
- sensory dissonance
- transient density
- harmonicity / inharmonicity
- binaural-beat pressure
- beat-conflict pressure
- tritone tension
- wolf-fifth tension

## Binaural Beats

For stereo material, USG estimates binaural-beat pressure from near-partial disagreements between left and right spectra. For mono material, it falls back to a conservative close-partial estimator.

That lets the analyzer respond to discomfort that is not well described by plain clipping or broadband harshness alone.

## Harmonicity, Inharmonicity, Tritones, And Wolf Fifths

The psycho profile explicitly gives ugliness credit to spectra that frustrate stable harmonic interpretation.

Qualitatively, ugliness tends to rise when:

- partials beat in the roughness zone
- interval content clusters around unstable or culturally tense intervals
- harmonic support weakens while inharmonic energy rises
- stereo channels disagree in ways that create binaural pressure

## Main Psycho Equation

```text
weighted_sum =
  -4.05
  + 1.60 * Phi_clip
  + 1.30 * Phi_rough
  + 1.00 * Phi_sharp
  + 1.00 * Phi_dissonance
  + 1.20 * Phi_transient
  + 0.90 * Phi_harsh
  + 1.25 * Phi_inharm
  + 0.85 * Phi_binaural
  + 1.05 * Phi_beatconflict
  + 0.85 * Phi_tritone
  + 0.75 * Phi_wolf
  - 0.45 * Phi_harmonicity
```

```text
colbys_psycho = clamp((2000 * sigmoid(weighted_sum)) - 1000, -1000, 1000)
```

## Joke Appendix: UglierBasis

`--joke` is intentionally separate from the real score.
It is decorative and optional.

```text
UglierBasis(x) = 1000 * sigmoid(Phi_1 + Phi_2 + Phi_3 + Phi_4 + Phi_5 + Phi_6 - lambda * Phi_7)
```

Component sketch:

- `Phi_1`: clip arrogance
- `Phi_2`: roughness
- `Phi_3`: sharpness
- `Phi_4`: dissonance
- `Phi_5`: transient density
- `Phi_6`: modulation bureaucracy across mixed terms
- `Phi_7`: harmonicity relief

Coefficient family:

- `alpha`: clip-pressure scaling
- `beta`: roughness scaling
- `gamma`: dissonance scaling
- `delta`: transient scaling
- `kappa`: mixed-term bureaucracy weights
- `lambda`: harmonicity relief weight

## References

Selected references that informed the repo's direction:

- Helmholtz, H. L. F. *On the Sensations of Tone*.
- Sethares, W. A. *Tuning, Timbre, Spectrum, Scale*.
- Zwicker, E., & Fastl, H. *Psychoacoustics: Facts and Models*.
- Harrison, P. M. C., & Pearce, M. T. (2020). *Simultaneous Consonance in Music Perception and Composition*.
- Kameoka, A., & Kuriyagawa, M. classic roughness / dissonance papers.
- Vassilakis, P. N. roughness and amplitude-fluctuation work.
- Parncutt, R. *Harmony: A Psychoacoustical Approach*.
- Eddins, D. A., Kopf, L. M., & Shrivastav, R. (2015). *The psychophysics of roughness applied to dysphonic voice*.
- Pickford, R. W. (1969). *The Psychology of Ugliness*.

## Limits

What this model does well:

- compare outputs inside the same repo/version
- steer search and mutation workflows
- make psychoacoustic tensions visible in CLI output

What it does not yet do:

- replace a listening study
- claim calibrated perceptual intervals in Colbys across populations
- serve as a standards-track measurement system
