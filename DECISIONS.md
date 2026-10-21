# DECISIONS.md — every nontrivial design choice and its rationale

Format: **decision** — *why* — *what was rejected*. Defaults are stated with their units.

---

## 1. Algorithm family

**Local, subset-based DIC with affine shape functions.** The prompt requires a CPU command-line
tool that works on an arbitrary directory of speckle images with no FE model, no training data
and no GPU, and that can mask regions and report per-point quality. Subset-based DIC is the only
family that satisfies all of these: it is embarrassingly parallel, isolates per-point failures,
masks trivially, and is what Ncorr, DICe, OpenCorr and py2DIC all implement.
*Rejected:* global/FE-DIC (needs a mesh, a global solve, and is less forgiving on unknown input),
Fourier/LSA DIC (needs a stationary pattern and struggles with arbitrary masks), learning-based
DIC (out of scope — no GPU/training; not clearly superior on smooth fields; see `RESEARCH.md` §2.5).

**Correlation criterion: ZNSSD.** Invariant to an affine illumination change (offset + gain),
which is the dominant nuisance in real tests, and it is the criterion the standard IC-GN
formulation assumes. *Rejected:* SSD (not illumination invariant), NSSD (invariant to gain only).

**Optimiser: inverse-compositional Gauss–Newton (IC-GN).** The Hessian and steepest-descent
images are computed once per point and reused for every frame and every iteration; IC-GN is
strictly cheaper than FA-NR at equal or better accuracy and noise robustness (Pan & Wang 2016).
*Rejected:* FA-NR (recomputes the Hessian per iteration), Levenberg–Marquardt (more robust in
theory but slower and unnecessary once the initial guess is good).

**Shape function: affine (6 parameters).** Captures translation, rotation, stretch and shear
inside the subset, and is *exact* for affine deformation (the provided dataset). Second-order
shape functions lower the bias for strongly non-uniform fields but double the parameter count and
slow convergence; not worth it here. Configurable via the code but not exposed on the CLI.

## 2. Interpolation

**Prefiltered B-spline of order 3 (bicubic), with an optional order 5 (biquintic).**
Bicubic *B-spline* has far less interpolation bias than a bicubic *polynomial* (iDICs GPG).
Gradients are obtained **exactly** from the B-spline derivative identity
`B_n'(x) = B_{n−1}(x+½) − B_{n−1}(x−½)` rather than by finite differences.
Order 5 is available (`--interp-order 5`) and is more accurate in the interior (~10⁻¹⁰ vs ~10⁻⁷
on an analytic test) but has a larger boundary effect (the quintic spline's mirror boundary
condition is less accurate), so **3 is the default**.
*Rejected:* bicubic polynomial (larger bias), Fourier interpolation (ringing), nearest/linear
(too coarse).

**Gaussian pre-filter, σ = 1.0 px, on by default.** The synthetic and many real speckle patterns
have sharp edges that are not band-limited, so the interpolant cannot reproduce the signal
between samples and the residual error is dominated by interpolation (aliasing) bias. Measured on
a synthetic 0.7-px-softness speckle with σ_grey = 2: subset 31, step 8, zero noise,
RMS spatial scatter 0.046 px unfiltered vs 0.037 px with σ = 1 px. iDICs treats "whether you
pre-filtered" as a reporting requirement, so the value is written into `params.json`.
*Rejected:* no pre-filter (larger bias), larger σ (loses spatial resolution and lowers the SSSIG,
raising the noise floor).

## 3. Initial guess and large motion

**Temporal prediction (linear extrapolation of the previous frames' parameters) by default.**
For smooth sequences the increment is sub-pixel and IC-GN converges in a few iterations with no
accumulation error, because every frame is correlated against the *original* reference.
*Rejected:* a per-point integer search on every frame (robust but ~10× slower), always running the
coarse pyramid (unnecessary cost).

**Hierarchical multi-scale affine fallback with a rotation search.** Triggered when the temporal
guess's median ZNCC < 0.95 (`init_fallback_zncc`), or on the first deformed frame. A brute-force
integer search at the coarsest pyramid level (factor up to 2⁴, ±24 coarse px), then ±2 px
refinement, with a robust affine fit. Because template matching decorrelates under large rigid
rotation, the search is repeated over candidate angles in steps of 10° up to ±45° and the best
score wins. The coarse guess is accepted only if it beats the temporal guess, so the fallback
cannot make the result worse. Validated on a single-step 30° rotation (u RMS 0.017 px).

**Reference strategy: `first` by default, `incremental` opt-in.** Incremental reference updating
accumulates interpolation and systematic error; the literature recommends updating only when
necessary. The default never accumulates. `--reference incremental --update-interval N` is
provided for genuinely decorrelating sequences and is documented as accumulating error.

## 4. ROI / masking

**Automatic detection by local grey-level standard deviation + Otsu/fraction threshold +
morphology + connected components.** Speckle is textured, background is flat, so local variance
separates them cleanly and cheaply. A unimodal texture test (`p5 > 0.5·p99`) detects a fully
speckled frame and keeps the whole image — important because a fully speckled frame is the common
case and must not be split arbitrarily. A user mask (255 = valid) overrides it.
*Rejected:* semantic segmentation (needs training data/GPU, no accuracy guarantee for the ROI that
matters), simple global thresholding (fails under vignetting), manual-only masking (the prompt
requires automatic detection).

**Correlation points require the *entire* subset footprint to be inside the ROI** (the mask is
eroded by `subset/2`). This keeps the Hessian identical for every point (no partial subsets, no
per-point weighted least squares) at the cost of a border of missing data near the ROI edge — the
standard trade-off. `--mask-mode` and `--mask-threshold` are configurable.

**Masker assumptions and failure modes** (also in the `mask.py` docstring): the detector
separates *textured from flat*, so a textured but non-specimen background (a printed grid, a
patterned fixture) is kept; a very smooth specimen region is discarded; a frame with no flat
background at all is detected as unimodal and kept whole; strong illumination gradients can bias
the local-variance map, in which case raise `--mask-window` or supply `--mask`.

## 5. Strain

**Local plane fit (virtual strain gauge) on the displacement grid.** Transparent, standard
(DICe's VSG), and the fitting window maps directly onto the well-understood spatial-resolution /
noise trade-off `L_VSG = (window−1)·step + subset`. On a regular grid the fit is a fixed
convolution, so it is fast. Default window 5×5 grid points.
*Rejected:* differentiating the displacement with finite differences (far noisier),
using the affine shape-function coefficients directly (available in the NPZ, but noisier and
harder to control the gauge length), spline fits (more complex, similar behaviour).

**Nominal (Biot) strain is the primary measure, with true (logarithmic) strain alongside it.**
The primary tensor is `ε_N = V − I = (F Fᵀ)^(1/2) − I`, whose principal values are the
engineering strains `λ_i − 1` and whose principal directions are in the current configuration.
This is exactly Abaqus output variable `NE` (“nominal strain”), so the fields can be compared
with an FE model. Unlike the linearised `sym(F) − I`, it vanishes under a finite rigid rotation
and remains valid at large deformation. The true (logarithmic/Hencky) strain `T = ln(V)` is also
computed and plotted next to it for comparison; `ln(V)` is the matrix logarithm, so its principal
values are `ln(λ_i)`.

**Equivalent strain assumes plane-strain incompressibility** (`ε_zz = −(ε_xx+ε_yy)`) — stated
explicitly because the alternative (plane stress) needs a Poisson ratio the tool does not know.
It is evaluated on the nominal components; for true strain the same plane-strain relation is
exact for isochoric deformation (`Etr_zz = −(Etr_xx+Etr_yy)`), so `Etr_eq` uses the same formula on the
true components.

**Sign convention (documented in the README):** x right, y down; `Eeng_xx = V₁₁ − 1`,
`Eeng_yy = V₂₂ − 1`, `Eeng_xy = V₁₂`, `Eeng_gamma_xy = 2 V₁₂`. At small strain these reduce to
`∂u/∂x`, `∂v/∂y`, `∂u/∂y + ∂v/∂x`. On the provided uniaxial-stretch dataset the nominal and
generator engineering
components coincide exactly (verified: Eeng_xx error ~5×10⁻⁴).

## 6. Uncertainty

**Per-point σ from `σ_r² H⁻¹`.** This is the standard linear-least-squares covariance of the
IC-GN parameters and reproduces the classic "noise / SSSIG" scaling. It is cheap (the Hessian
inverse already exists) and gives a spatially varying map, which a single global number cannot
(the noise floor is spatially heterogeneous; Peshave et al. 2025).

**Image noise from the DIC matching residual, corrected for the pre-filter.** More robust for
sharp speckle than Immerkær's Laplacian estimator (which is contaminated by pattern content and
overestimated the provided dataset's σ = 2 as 5.2). Measured 2.36 on `Data01/` (true 2.0).
Immerkær is kept as a fallback when no correlation residual is available.

**Empirical zero-motion noise floor.** The only assumption-free way to tell the user what their
data can resolve; recommended by iDICs. Implemented by correlating two independently noisy copies
of the user's own reference image with the user's own settings.

## 7. Numerical defaults

| Setting | Default | Rationale |
|---|---|---|
| `--subset` | 41 px | ≥ 3 speckles of 3–6 px for typical patterns; good bias/noise balance (measured: subset 41 gives 0.025 px RMS vs 0.19 px at 21 px on a 0.7-softness synthetic) |
| `--step` | 16 px | ≈ subset/2.5; dense enough to sample gradients, keeps the point count and runtime moderate |
| `--strain-window` | 5 points | VSG ≈ 105 px at the defaults; halves the strain noise vs 3 while keeping the gauge local |
| `--interp-order` | 3 | see §2 |
| `--prefilter` | 1.0 px | see §2 |
| `--min-zncc` | 0.5 | a genuine match on speckle is > 0.9; 0.5 is a conservative "something is wrong" floor |
| `--min-coverage` | 0.30 | fail loudly below this (R16); the user can lower it |
| `--max-iter` | 30 | typical convergence is 3–8; 30 bounds the cost of pathological points |
| `--conv-tol` | 1e-4 | px for translation, dimensionless for strain; below the noise floor |
| `--init-fallback-zncc` | 0.95 | temporal prediction is near-perfect on smooth sequences; below this the coarse search is worth its cost |
| `--dt` | 1.0 s | the prompt says frames are at a constant interval; 1.0 makes the time axis the frame count if the user does not override it |
| `--scale` | 1.0 px/px | dimensionless by default; the user supplies mm/px |

## 8. Libraries and licenses

| Library | Min version | License | Why |
|---|---|---|---|
| numpy | 1.24 | BSD-3 | arrays, linear algebra |
| scipy | 1.10 | BSD-3 | `ndimage` (B-splines, morphology, filters), `spatial.cKDTree`, `interpolate` |
| pillow | 9.5 | HPND (MIT-like) | reading/writing 8/16-bit PNG/TIF/JPEG/BMP |
| matplotlib | 3.7 | PSF-based (BSD-compatible) | figures and animations |
| tqdm | 4.65 | MPL-2.0 / MIT | *optional* progress bar; the code falls back to logging |
| pytest, ruff | 7.4 / 0.4 | MIT | *dev only* |

No JIT compilation (numba/Cython) is used, so there is no optional accelerator to guard. The
vectorised NumPy implementation meets the performance target (≈2.4 s/frame for 3 844 points on
1024²). No code is copied from Ncorr, DICe, OpenCorr or py2DIC; the algorithms are implemented
from the literature (`RESEARCH.md` §9). BSD-licensed code *could* have been reused, but a clean,
understood implementation is easier to validate and document.

## 9. Interface and output choices

* **Two positional arguments** (images, output) as required; every other setting is an option
  with a documented default and unit, so `--help` is the reference.
* **Exit codes**: 0 ok, 1 unexpected, 2 usage (argparse), 3 input problem, 4 coverage collapse,
  5 validation failure. Distinct codes let a script branch on the failure mode.
* **Invalid points are `NaN`, never interpolated or fabricated.** Reported as `NaN` in every NPZ
  and CSV, and excluded from every statistic (R16).
* **Fields are overlaid on the current (deformed) image by default, with the field placed at the
  material positions `X + u`.** The correlation grid lives in the reference frame, but drawing the
  field on the *undeformed* image for every frame makes the specimen look frozen: the contours
  move but the picture behind them does not, which is misleading in an animation of a deforming
  test. Placing the field at `X + u` on the current frame's image gives a Lagrangian (material)
  presentation in which the specimen visibly deforms, the field follows the material, and the
  material leaving the field of view is visible. `--overlay reference` restores the classic
  fixed-reference presentation (undeformed image, field at the reference positions), which is the
  better choice when several frames must be compared in one common coordinate system.
  *This default was changed after a user reported that the animation's background never moved;
  the earlier choice was wrong for a per-frame/animation presentation.*
* **The ROI boundary is drawn in whichever frame is used**: in `reference` mode as the mask
  contour, in `deformed` mode as the mask boundary warped by the measured displacement field, so
  requirement R10 ("mask shown") holds in both presentations.
* A quiver overlay on the `|u|` panel shows the motion direction, with arrows starting at the
  material points.
* **Nominal and true (logarithmic) strains are plotted side by side** so the two finite-strain
  measures can be compared directly.
* **Machine-readable output is the primary product**, figures the secondary: per-frame NPZ with
  the displacement, quality and strain arrays, per-frame time-series CSVs, a grid CSV and a
  `params.json` with versions and input hashes (reproducibility, §5.8).
* **Extra visualizations implemented (R14):** mean/max displacement and strain versus time
  (with the per-point uncertainty and the measured noise floor), horizontal/vertical line
  profiles, an uncertainty/quality map, and growing-field animations for `|u|` and `ε_xx`.
  These are the tools a mechanical-testing engineer actually uses; nothing decorative was added.
* **Animations default to GIF** so that no ffmpeg installation is required; `--animation-format mp4`
  is available and falls back to GIF if ffmpeg is missing.

## 10. Reproducibility

* Fixed seeds for every stochastic step (noise floor, synthetic generation); documented in
  `params.json`.
* Deterministic point ordering (natural sort, row-major grid), so the same input and settings give
  byte-identical NPZ arrays (tested).
* `params.json` records the package versions, the exact configuration, and the SHA-256 of every
  input file.

## 11. Discrepancies and honest notes

* **The prompt describes `Data01/` as 40 images; the directory actually contains 100**
  (`image0001.png` … `image0100.png`, with 100 masks and 100 ground-truth rows). The tool reads
  whatever is in the directory, so all 100 are processed. The prompt's description of the
  deformation (uniform uniaxial stretch to ≈40 % nominal strain, additive Gaussian noise
  σ ≈ 2 grey levels, no rotation, no translation, outer field of view leaving the frame late in
  the sequence) matches the actual data.
* **Order-5 interpolation** is offered but is not the default: its interior accuracy is better
  but its boundary behaviour is worse, and boundary subsets are common in this application.
* **The single-step 30° rotation** case was initially failing (u RMS 5.2 px) because template
  matching decorrelates under a 30° rotation and the coarse search was translation-only. It was
  fixed by adding the rotation search described in §3; it now passes at 0.017 px. This is
  recorded rather than hidden.
* **Incremental mode** is implemented and unit-tested but is not the recommended default; its
  accuracy is not claimed to be better than the direct mode and it accumulates error by design.
* **Speed** is measured, not estimated: see `validation_report.md`.
