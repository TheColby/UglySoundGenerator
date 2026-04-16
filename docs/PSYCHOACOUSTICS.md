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

```math
\mathrm{weighted\_sum} = -4.05 + 1.6\Phi_{\mathrm{clip}} + 1.3\Phi_{\mathrm{rough}} + 1.0\Phi_{\mathrm{sharp}} + 1.0\Phi_{\mathrm{dissonance}} + 1.2\Phi_{\mathrm{transient}} + 0.9\Phi_{\mathrm{harsh}} + 1.25\Phi_{\mathrm{inharm}} + 0.85\Phi_{\mathrm{binaural}} + 1.05\Phi_{\mathrm{beatconflict}} + 0.85\Phi_{\mathrm{tritone}} + 0.75\Phi_{\mathrm{wolf}} - 0.45\Phi_{\mathrm{harmonicity}}
```

```math
\mathrm{colbys}_{\mathrm{psycho}} = \mathrm{clamp}\!\left(2000\,\sigma(\mathrm{weighted\_sum}) - 1000, -1000, 1000\right)
```

## Joke Appendix: UglierBasis

`--joke` is intentionally separate from the real score.
It is decorative and optional.

```math
\mathfrak{U}_{\mathrm{UglierBasis}}(x)=1000\,\sigma\!\left(\Phi_{1}+\Phi_{2}+\Phi_{3}+\Phi_{4}+\Phi_{5}+\Phi_{6}-\lambda\Phi_{7}\right)
```

Component sketch:

- $\Phi_1$: clip arrogance
- $\Phi_2$: roughness
- $\Phi_3$: sharpness
- $\Phi_4$: dissonance
- $\Phi_5$: transient density
- $\Phi_6$: modulation bureaucracy across mixed terms
- $\Phi_7$: harmonicity relief

Coefficient family:

- $\alpha$: clip-pressure scaling
- $\beta$: roughness scaling
- $\gamma$: dissonance scaling
- $\delta$: transient scaling
- $\kappa$: mixed-term bureaucracy weights
- $\lambda$: harmonicity relief weight

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
