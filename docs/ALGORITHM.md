# ALGORITHM.md — the mathematics of the implementation

Notation: images are indexed `[row = y, col = x]`; `x` increases to the right, `y` increases
downwards. `f` is the reference (undeformed) image, `g` the deformed image. A point of interest
(POI) has reference coordinates `x₀ = (x₀, y₀)` and local subset coordinates
`ξ = (ξ_x, ξ_y) = (x − x₀, y − y₀)`.

---

## 1. Kinematic model: affine warp

The first-order (affine) warp maps the reference subset to the deformed image:

```
        ⎛ X_t ⎞   ⎛ x₀ ⎞   ⎛ u ⎞   ⎡ 1+u_x   u_y ⎤ ⎛ ξ_x ⎞
        ⎜     ⎟ = ⎜    ⎟ + ⎜   ⎟ + ⎢             ⎥ ⎜     ⎟
        ⎝ Y_t ⎠   ⎝ y₀ ⎠   ⎝ v ⎠   ⎣  v_x   1+v_y ⎦ ⎝ ξ_y ⎠
```

with the six parameters `p = (u, v, u_x, u_y, v_x, v_y)`. Writing `A` for the 2×2 block and `t`
for `(u, v)`, the warp is `W(ξ; p) = A ξ + t`.

**Why affine.** It captures translation, rotation, normal and shear strain inside a subset. For a
globally affine deformation (as in the provided dataset and most of the validation cases) it is
*exact*, so the only residual errors are interpolation bias and image noise. iDICs and the DIC
Challenge reference analyses treat affine + B-spline as the standard configuration.

---

## 2. Matching criterion: ZNSSD

```
C_ZNSSD(p) = Σ_ξ [ (f(ξ) − f̄)/Δf − (g(W(ξ;p)) − ḡ)/Δg ]²
f̄ = mean(f),  Δf = √(Σ_ξ (f − f̄)²)      (same for g)
```

ZNSSD is invariant to an affine change of the illumination (`g → a·g + b`), which is the dominant
nuisance in real tests, and it is the criterion for which the IC-GN Hessian is naturally
formulated. The equivalent zero-normalised cross-correlation is `ZNCC = 1 − C_ZNSSD/2`, used here
as the per-point quality measure (`ZNCC = 1` is a perfect match).

---

## 3. Optimisation: inverse-compositional Gauss–Newton (IC-GN)

**Why IC-GN.** In forward-additive Gauss–Newton the Jacobian and Hessian depend on the *current*
warp, so they must be recomputed at every iteration for every point. In the inverse-compositional
formulation the increment is applied to the *reference* image, so the Jacobian is evaluated once
at `p = 0` and never changes; only the deformed-image sampling is repeated. Pan & Wang (2016)
showed IC-GN matches or beats FA-NR in accuracy and noise robustness at a fraction of the cost.

**Reference-side precomputation** (once per POI, reused for every frame):

```
∇f = (f_x, f_y)                                  (exact B-spline derivatives, §4)
SD = [ f_x,  f_y,  f_x ξ_x,  f_x ξ_y,  f_y ξ_x,  f_y ξ_y ]      (steepest-descent images)
H  = Σ_ξ SD SDᵀ / Δf²                            (6×6, constant)
```

`SD_i = ∇f · ∂W/∂p_i` with

```
∂W/∂u = (1,0)ᵀ   ∂W/∂v = (0,1)ᵀ
∂W/∂u_x = (ξ_x,0)ᵀ  ∂W/∂u_y = (ξ_y,0)ᵀ
∂W/∂v_x = (0,ξ_x)ᵀ  ∂W/∂v_y = (0,ξ_y)ᵀ
```

**Iteration.** At iteration *k* with the current warp `p_k`:

```
g_norm = (g(W(ξ;p_k)) − ḡ)/Δg
r      = f_norm − g_norm                     (normalised residual)
Δp     = −H⁻¹ Σ_ξ SD · r / Δf
p_{k+1} = p_k ∘ Δp⁻¹
```

The inverse composition of two affine warps is closed-form:

```
A_new = A_old A_Δ⁻¹ ,     t_new = t_old − A_new t_Δ
```

Convergence is declared when `max(|Δu|, |Δv|) < tol` **and** `max(|Δu_x|, …, |Δv_y|) < tol`; the
default `tol = 1e-4` (px for the translation, dimensionless for the strain parameters).

---

## 4. Sub-pixel interpolation: prefiltered B-splines with exact derivatives

`f` and `g` are sampled at arbitrary sub-pixel positions with a prefiltered B-spline of order
`n` (3 = bicubic, 5 = biquintic) evaluated by `scipy.ndimage.map_coordinates` on the spline
coefficient array (`prefilter=False`).

The gradient is obtained **exactly**, using the B-spline identity

```
d/dx B_n(x) = B_{n−1}(x + ½) − B_{n−1}(x − ½)
```

so that

```
∂/∂x Σ_k c_k B_n(x−k) = Σ_k c_k [B_{n−1}(x−k+½) − B_{n−1}(x−k−½)]
```

which is the difference of two order-`n−1` evaluations on the *same* coefficients. No
finite-difference approximation is involved. (Verified against analytic functions in
`tests/test_interpolation.py`: order 3 reproduces values to ~5×10⁻⁷ and gradients to ~1×10⁻⁴ in
the interior; order 5 to ~1×10⁻¹⁰ and ~1×10⁻⁵.)

A Gaussian pre-filter (`--prefilter`, default σ = 1 px) is applied to both images before
correlation. The synthetic speckle has sharp edges (speckle softness 0.7 px) that are not
band-limited, so interpolation bias dominates the error budget; the pre-filter removes the
out-of-band content and measurably reduces both bias and variance (iDICs Good Practices Guide).

---

## 5. Initial guess and large motion

IC-GN is a local method: it converges to the nearest local minimum of ZNSSD, so the initial guess
must lie in the right basin. Three mechanisms, in order:

1. **Temporal prediction.** `p₀(frame i) = 2 p(frame i−1) − p(frame i−2)` (linear extrapolation;
   the previous frame's value for the first deformed frame). For smooth sequences this is
   sub-0.1 px accurate and IC-GN converges in 3–6 iterations with no accumulation error, because
   every frame is still correlated against the **original reference**.
2. **Fallback trigger.** If the median ZNCC of the temporal guess is below
   `init_fallback_zncc` (default 0.95), the coarse search runs.
3. **Hierarchical multi-scale affine search.** Both images are downsampled by `2^level`
   (anti-aliased) and a grid of points inside the ROI is matched by brute-force integer search
   (±`coarse_search` px at the coarsest level, ±2 px afterwards); a robust affine is fitted with
   iterative outlier rejection. Because template matching decorrelates under large rigid
   rotation, the search is repeated over candidate rotation angles (±`coarse_angle_max` in
   `coarse_angle_step` steps) and the angle with the best median ZNCC wins. The coarse guess is
   accepted only if it beats the temporal guess, so the fallback can never make things worse.

**Reference strategy.** `--reference first` (default) correlates every frame to the original
reference: no accumulation, and it works as long as the pattern has not decorrelated.
`--reference incremental` promotes the reference every `--update-interval` frames and accumulates
the total displacement and deformation gradient:

```
d_total(X) ← d_total(X) + Δd(x_ref)
F_total    ← F_inc F_total            (equivalently H_total = H_ref + H_inc (I + H_ref))
```

Accumulation of interpolation and systematic error is the price; the literature recommends
updating only when necessary, which is what `--update-interval` controls.

---

## 6. Reliability-guided re-seeding

After the first IC-GN pass a point is flagged *suspicious* if it is invalid, if its ZNCC is below
`reseed_zncc` (0.9), or if it disagrees with the local median of its `k` nearest neighbours by
more than `max(outlier_px, 4 × local_spread)`, where `local_spread` is the MAD of the neighbours'
displacements (this makes the test scale-aware, so a strongly graded field is not mistaken for a
field of outliers). Suspicious points are restarted from the neighbour median and IC-GN is run
again; a new solution replaces the old one only if its ZNCC is higher. Points that remain poor
are left invalid. This is the RG-DIC idea of Ncorr, simplified.

---

## 7. Region of interest (masking)

1. Compute the local grey-level standard deviation `σ_I(x)` with a `--mask-window` box filter.
2. If `p5(σ_I) > 0.5 · p99(σ_I)` the texture is unimodal — there is no flat background — and the
   whole frame is kept.
3. Otherwise threshold with Otsu's method when it produces a plausible split (2–90 % of the
   frame), else with `p5 + f · (p99 − p5)` where `f = --mask-threshold`.
4. Morphological closing then opening (disk of `morphology_radius`), hole filling, and
   connected-component filtering (keep components ≥ `min_component_area` and
   ≥ `min_component_fraction` of the largest).
5. The correlation grid only accepts points whose **entire** subset footprint lies inside the
   mask (`make_grid` erodes by `subset/2`), which keeps the Hessian identical for every point.

A user mask (`--mask`, 255 = valid) overrides the automatic detection.

---

## 8. Strain: virtual strain gauge

For each scalar displacement component the field is fitted locally with a first-order plane
`z = a₀ + a₁ X + a₂ Y` over a `window × window` block of grid samples. On a regular grid this is a
fixed convolution whose kernels are the first two rows of the pseudo-inverse of the design matrix
`[1, X, Y]`, so the whole field is computed in one pass.

The displacement gradient and the deformation gradient (image frame, x right, y down):

```
H = ⎡ u_x  u_y ⎤        F = I + H
    ⎣ v_x  v_y ⎦
```

The **primary strain measure is the nominal (Biot/engineering) strain**

```
ε_N = V − I = (F Fᵀ)^(1/2) − I
V = left stretch tensor from the polar decomposition F = V R
```

with components

```
Eeng_xx = V₁₁ − 1,   Eeng_yy = V₂₂ − 1,
Eeng_xy = V₁₂,       Eeng_gamma_xy = 2 V₁₂
```

The principal values of `ε_N` are `λ_i − 1`, i.e. the engineering strains, and the principal
directions are in the **current** configuration. This is the strain measure Abaqus reports as
`NE` (“nominal strain”), so the output is directly comparable with an FE model. A rigid rotation
gives `V = I`, hence `ε_N = 0`; a uniaxial stretch `λ` gives `λ − 1`.

The **true (logarithmic/Hencky) strain** is also reported:

```
T = ln(V)
```

where `ln(V)` is the **matrix logarithm** of the symmetric positive-definite left stretch
tensor—not an element-wise logarithm. Its components are

```
Etr_xx = (ln V)₁₁,   Etr_yy = (ln V)₂₂,   Etr_xy = (ln V)₁₂,   Etr_gamma_xy = 2 Etr_xy
```

The principal values of `T` are `ln(λ_i)` and its principal directions are in the current
configuration. A rigid rotation gives `T = 0`, and a uniaxial stretch `λ` gives `Etr_xx = ln λ`.
The matrix logarithm is evaluated in closed form from the symmetric eigen-decomposition of `V`.
Writing `V = mean·(I + E)` with `E = (V − mean·I)/mean` traceless and
`δ = (λ₁ − λ₂)/(λ₁ + λ₂) = rad(V)/mean(V)`,

```
ln(V) = ½ ln(det V) · I + (atanh(δ)/δ) · E
```

The coefficient uses its limit 1 when `δ → 0` (equal principal stretches), which makes the
construction numerically stable for nearly equal eigenvalues.

Also reported: small-angle rotation `ω = ½(v_x − u_y)`, finite rotation angle from the polar
decomposition `R = F (FᵀF)^{-1/2}` (closed form for 2×2), principal nominal and true strains,
and von Mises equivalent strains. For the nominal components the small-strain plane-strain
relation `Eeng_zz = −(Eeng_xx + Eeng_yy)` is used:

```
Eeng_eq = √( (2/3) [ Eeng_xx² + Eeng_yy² + (Eeng_xx+Eeng_yy)² + 2 Eeng_xy² ] )
```

For true strain, isochoric plane strain gives `Etr_zz = −(Etr_xx + Etr_yy)` exactly, so the same
formula is applied directly to `Etr_xx`, `Etr_yy`, `Etr_xy`:

```
Etr_eq = √( (2/3) [ Etr_xx² + Etr_yy² + (Etr_xx+Etr_yy)² + 2 Etr_xy² ] )
```

**Spatial resolution.** The reported strain is an average over the virtual strain gauge length
`L_VSG = (window − 1)·step + subset` (iDICs Good Practices Guide, Ch. 5); a strain peak narrower
than `L_VSG` is attenuated. Strain noise scales roughly as `σ_ε ≈ σ_u / L_VSG`.

---

## 9. Uncertainty

**Per point (model based).** The IC-GN parameter covariance follows from linear least squares on
the normalised residual:

```
Cov(Δp) = σ_r² (JᵀJ)⁻¹ ,      J = SD / Δf
        = σ_r² H⁻¹
σ_u = σ_r √(H⁻¹₀₀),   σ_v = σ_r √(H⁻¹₁₁)
```

with `σ_r` the standard deviation of the normalised matching residual `f_norm − g_norm`. The
units work out to pixels. This reproduces the classic result that the random displacement error
is proportional to the image noise and inversely proportional to the sum of squared intensity
gradients in the subset (Wang et al. 2015). `σ_u`, `σ_v` and the residual are written into every
`fields/frame_XXXX.npz`.

**Image noise.** Estimated from the DIC matching residual itself:
`σ_grey = median(σ_r · Δf) / (√2 · gain)`, where `gain` is the noise-attenuation factor of the
Gaussian pre-filter and the `√2` accounts for two independently noisy images. This is more robust
for a sharp speckle pattern than the Immerkær Laplacian estimator (which is kept as a fallback).

**Noise floor.** The pipeline is run on a synthetic pair of *zero-motion* images carrying the
measured noise, and the apparent displacement and strain are measured. This is the
iDICs-recommended noise-floor analysis and it needs no ground truth. It is the number the user
should compare their features against.

---

## 10. Complexity and performance

Let `N` be the number of POIs and `S = subset²` the number of samples per subset. Per frame:

* IC-GN: `O(I · N · S)` with `I ≈ 3–8` iterations, fully vectorised over `N`.
* Reference precomputation: `O(N · S)` once for the whole sequence.
* Strain: `O(N · window²)` via convolution.

Measured throughput is reported in `validation_report.md`. The per-point work is dominated by
`map_coordinates`, which is C code; the outer loop over frames is sequential in the default mode
because the temporal initial guess depends on the previous frame.

---

## References

See `docs/RESEARCH.md` §9 for the full list. The equations above follow Sutton, Orteu &
Schreier (2009), Pan et al. (2013, IC-GN), Pan & Wang (2016), Wang et al. (2015, uncertainty),
Turner (2018, virtual strain gauge) and the iDICs Good Practices Guide (2018/2025).
