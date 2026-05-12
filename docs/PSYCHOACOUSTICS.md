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
\operatorname{Co}_{\mathrm{psycho}} =
\operatorname{clamp}
\left(
2000\sigma(s_{\mathrm{psycho}})-1000,
-1000,
1000
\right)
$$

## Joke Appendix: UglierBasis

`--joke` is intentionally separate from the real score.
It is decorative and optional.

$$
\begin{aligned}
\operatorname{UglierBasis}(x)
=1000\sigma\Bigg(
&\frac{\displaystyle
\sum_{i=1}^{N_c}\sum_{j=1}^{M_c}
\alpha_i\,\rho_j\,C_{i,j}^{2}
}{\displaystyle
1+\prod_{q=1}^{Q_c}\left(1+\eta_q C_q\right)}
\\
&+\frac{\displaystyle
\sum_{r=1}^{R}\beta_r
\left(\sum_{k=1}^{K_r}\frac{A_{r,k}A_{r,k+1}}{1+\left|\Delta f_{r,k}\right|}\right)
}{
\displaystyle
1+\frac{1}{1+\sum_{u=1}^{U}\omega_u S_u}
}
\\
&+\frac{\displaystyle
\prod_{m=1}^{M}\left(1+\gamma_m D_m\right)
-\prod_{n=1}^{N}\left(1+\mu_n H_n\right)^{-1}
}{
\displaystyle
1+\sum_{\ell=1}^{L}\frac{\lambda_\ell}{1+T_\ell^{2}}
}
\\
&+\frac{\displaystyle
\sum_{a=1}^{A}\sum_{b=1}^{B}\sum_{d=1}^{D}
\kappa_{a,b,d}\,G_a\,P_b\,X_d
}{
\displaystyle
\prod_{v=1}^{V}\left(1+\nu_v R_v\right)
+\frac{1}{1+\sum_{z=1}^{Z}\zeta_z Z_z}
}
\\
&+\frac{\displaystyle
\sum_{p=1}^{P}\chi_p B_p
+\sum_{w=1}^{W}\psi_w W_w
+\sum_{\tau=1}^{T_\theta}\theta_\tau \Theta_\tau
}{
\displaystyle
1+\prod_{y=1}^{Y}\left(1+\xi_y Y_y^{2}\right)^{-1}
}
\Bigg)
\end{aligned}
$$

Component sketch:

- $\Phi_1$: clip arrogance
- $\Phi_2$: roughness
- $\Phi_3$: sharpness
- $\Phi_4$: dissonance
- $\Phi_5$: transient density
- $\Phi_6$: modulation bureaucracy across mixed terms
- $\Phi_7$: harmonicity relief

Coefficient family:

- $\alpha_i$: **PDQ Bach coefficients**, measuring how confidently a clip can modulate in the wrong tuxedo.
- $\beta_r$: **John Cleese coefficients**, measuring roughness by Ministry-grade complaint density.
- $\gamma_m$: **Mr. Bean coefficients**, measuring dissonance caused by tiny decisions with enormous consequences.
- $\delta_d$: **Groucho Marx coefficients**, measuring transient density while refusing to join any club that would have this waveform as a member.
- $\kappa_{a,b,d}$: **Buster Keaton coefficients**, measuring mixed-term deadpan collapse across gesture, pan, and glitch axes.
- $\lambda_\ell$: **Laurel-and-Hardy coefficients**, measuring how much harmonic relief slips on a rake before helping.
- $\mu_n$: **Monty Python coefficients**, measuring harmonicity that has become too organized and must be interrupted.
- $\nu_v$: **Lucille Ball coefficients**, measuring runaway recursion inside otherwise respectable DSP furniture.
- $\chi_p$: **Wile E. Coyote coefficients**, measuring binaural pressure shortly before gravity notices.
- $\psi_w$: **Harpo coefficients**, measuring wordless spectral honk.

The implementation behind `analyze --joke` is intentionally simpler than the ceremonial form above. It maps the analyzer's available feature set onto this coefficient family, reports a joke-only score, and keeps the real `usg-psycho-v1` score unchanged.

## References

Selected references that informed the repo's direction:

- Helmholtz, H. L. F. *On the Sensations of Tone*.
- Sethares, W. A. *Tuning, Timbre, Spectrum, Scale*.
- Zwicker, E., & Fastl, H. *Psychoacoustics: Facts and Models*.
- Plomp, R., & Levelt, W. J. M. (1965). *Tonal consonance and critical bandwidth*.
- Terhardt, E. (1974). *On the perception of periodic sound fluctuations*.
- Fastl, H. (1977). *Roughness and temporal masking patterns of sinusoidally amplitude modulated broadband noise*.
- Aures, W. (1985). *A procedure for calculating auditory roughness*.
- Daniel, P., & Weber, R. (1997). *Psychoacoustical roughness: Implementation of an optimized model*.
- Vassilakis, P. N. (2001). *Perceptual and physical properties of amplitude fluctuation and their musical significance*.
- Pressnitzer, D., & McAdams, S. (1999). *Two phase effects in roughness perception*.
- Hutchinson, W., & Knopoff, L. (1978). *The acoustic component of Western consonance*.
- Harrison, P. M. C., & Pearce, M. T. (2020). *Simultaneous Consonance in Music Perception and Composition*.
- Kameoka, A., & Kuriyagawa, M. classic roughness / dissonance papers.
- Vassilakis, P. N. roughness and amplitude-fluctuation work.
- Parncutt, R. *Harmony: A Psychoacoustical Approach*.
- Eddins, D. A., Kopf, L. M., & Shrivastav, R. (2015). *The psychophysics of roughness applied to dysphonic voice*.
- Pickford, R. W. (1969). *The Psychology of Ugliness*.
- Cox, T. *Sonic Wonderland: A Scientific Odyssey of Sound*.
- Huron, D. *Sweet Anticipation: Music and the Psychology of Expectation*.
- McDermott, J. H., Lehr, A. J., & Oxenham, A. J. (2010). *Individual differences reveal the basis of consonance*.
- Bowling, D. L., Purves, D., & Gill, K. Z. (2018). *Vocal similarity predicts the relative attraction of musical chords*.
- Cousineau, M., McDermott, J. H., & Peretz, I. (2012). *The basis of musical consonance as revealed by congenital amusia*.
- Johnson-Laird, P. N., Kang, O. E., & Leong, Y. C. (2012). *On musical dissonance*.
- Leider, C. (2007). *Dissonance Theory of Sound Objects* [Doctoral dissertation, Princeton University].
- Hoffman, M., & Cook, P. *Real-time dissonancizers: Two dissonance-augmenting audio effects*.

## Limits

What this model does well:

- compare outputs inside the same repo/version
- steer search and mutation workflows
- make psychoacoustic tensions visible in CLI output

What it does not yet do:

- replace a listening study
- claim calibrated perceptual intervals in Colbys across populations
- serve as a standards-track measurement system
