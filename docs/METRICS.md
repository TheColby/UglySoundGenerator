# Metrics

USG uses one public metric: **Colbys (Co)**.

## Scale

$$
\mathrm{Colbys}(x) \in [-1000, 1000]
$$

Interpretation:

- `-1000`: cleanest / least ugly
- `0`: neutral center
- `+1000`: most ugly

## `go` Mapping

`usg go --level` is specified directly in Colbys.
The engine converts that to an internal normalized intensity for modulation strength:

$$
\mathrm{I}(c)=\mathit{clamp}\left(\frac{c-(-1000)}{2000},0,1\right)
$$

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

$$
\mathrm{Co}_{\mathrm{basic}} =
\mathit{clamp}
\left(
20\left(1.6C+45H+200Z\right)-1000,
-1000,
1000
\right)
$$

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

$$
\begin{aligned}
s_{\mathrm{psycho}}={}&-4.05
+1.60\Phi_{\mathrm{clip}}
+1.30\Phi_{\mathrm{rough}}
+1.00\Phi_{\mathrm{sharp}} \\
&+1.00\Phi_{\mathrm{dissonance}}
+1.20\Phi_{\mathrm{transient}}
+0.90\Phi_{\mathrm{harsh}}
+1.25\Phi_{\mathrm{inharm}} \\
&+0.85\Phi_{\mathrm{binaural}}
+1.05\Phi_{\mathrm{beatconflict}}
+0.85\Phi_{\mathrm{tritone}}
+0.75\Phi_{\mathrm{wolf}} \\
&-0.45\Phi_{\mathrm{harmonicity}}
\end{aligned}
$$

$$
\mathrm{Co}_{\mathrm{psycho}} =
\mathit{clamp}
\left(
2000\sigma(s_{\mathrm{psycho}})-1000,
-1000,
1000
\right)
$$

## Why The Versioning Matters

The repo now treats these weights as named profiles rather than invisible magic numbers. That gives downstream users a stable contract:

- scores can be compared only within the same profile version
- future tuning can ship as `usg-basic-v2` or `usg-psycho-v2`
- JSON consumers can store the profile name next to the score

## Normalization Note

USG normalizes by sample peak unless you disable normalization. That is useful and predictable, but it is not true-peak or loudness normalization.
