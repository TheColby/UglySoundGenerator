# Metrics

USG uses one public metric: **Colbys (Co)**.

## Scale

```text
Colbys in [-1000, 1000]
```

Interpretation:

- `-1000`: cleanest / least ugly
- `0`: neutral center
- `+1000`: most ugly

## `go` Mapping

`usg go --level` is specified directly in Colbys.
The engine converts that to an internal normalized intensity for modulation strength:

```text
I(c) = clamp((c - (-1000)) / 2000, 0, 1)
```

where `c` is the target Colbys value and `I(c)` is the internal drive intensity.

This keeps the public model simple while preserving a useful `0..1` control inside the DSP.

## Analysis Profiles

USG currently exposes two explicit, versioned heuristic profiles:

- `usg-basic-v1`
- `usg-psycho-v1`

Both are transparent heuristics, not listening-test-calibrated standards.
The analyzer reports that fact in JSON output via `score_metadata`.

## Basic Profile

The basic profile is a fast time-domain proxy built from:

- clipped-sample percentage
- harshness ratio
- zero-crossing rate

```text
colbys_basic = clamp(((1.6 * C) + (45 * H) + (200 * Z)) * 20 - 1000, -1000, 1000)
```

where:

- `C` = clipped sample percentage
- `H` = harshness ratio
- `Z` = zero-crossing rate

## Psycho Profile

The psycho profile works on top of FFT-derived features:

- roughness
- sharpness
- dissonance
- transient density
- harmonicity / inharmonicity
- binaural-beat pressure
- beat conflict
- tritone tension
- wolf-fifth tension

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

## Why The Versioning Matters

The repo now treats these weights as named profiles rather than invisible magic numbers. That gives downstream users a stable contract:

- scores can be compared only within the same profile version
- future tuning can ship as `usg-basic-v2` or `usg-psycho-v2`
- JSON consumers can store the profile name next to the score

## Normalization Note

USG normalizes by sample peak unless you disable normalization. That is useful and predictable, but it is not true-peak or loudness normalization.
