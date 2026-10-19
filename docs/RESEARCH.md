# State of the Art in 2D Digital Image Correlation — a research survey

**Deliverable of Phase 1** for the 2D-DIC application described in `Prompt2.md`.
Written before any application code, from real web searches. Every reference was opened and
checked; links and claims are listed in §9.

---

## 1. Executive summary

1. **Local subset-based DIC with an inverse-compositional Gauss–Newton (IC-GN) solver is the
   current default for a CPU command-line tool.** It matches or beats forward-additive
   Newton–Raphson (FA-NR) on accuracy and noise robustness while being an order of magnitude
   cheaper, because the Hessian and steepest-descent images are computed once per point from the
   reference image and reused at every iteration. Published sub-pixel noise floors for
   well-speckled images are ~0.01–0.05 px (Wang et al. 2015; Pan & Wang 2016; iDICs GPG 2018/2025).
2. **Zero-normalised sum of squared differences (ZNSSD) is the matching criterion of choice.**
   It is invariant to offset and scale of the illumination, which is the dominant nuisance in real
   tests, and it is the criterion assumed by the standard IC-GN formulation. SSD is only used in
   closed-form error analyses, not in the measurement itself.
3. **Affine (first-order) subset shape functions with a B-spline interpolant are the best
   accuracy/complexity trade-off.** Affine shape functions capture stretch and rotation within a
   subset and are exact for the affine ground truth used in this project; bicubic or biquintic
   B-splines give interpolation bias one to two orders of magnitude below pixel-level
   (`bicubic polynomial` is markedly worse; iDICs GPG 2025, Tip 5.6).
4. **Large deformation must be handled by reference updating, but *direct* correlation against the
   original reference is preferable whenever it still correlates, because incremental accumulation
   drifts.** The literature is explicit: incremental DIC (every frame becomes the reference)
   accumulates systematic interpolation error; conditional/limited reference updating reduces it
   substantially (incremental-DIC error analysis, 2023). Our design therefore correlates every
   frame back to the original reference using a temporal *initial guess* for speed, and offers
   incremental reference updating as an explicit option for cases that genuinely decorrelate.
5. **Strain should be computed by local plane (polynomial) fitting of the displacement field**
   (a virtual strain gauge). The strain-window size sets a hard spatial-resolution/noise
   trade-off; the effective gauge length is `L_VSG = (window−1)·step + subset` (iDICs GPG,
   Ch. 5). Reporting must include this trade-off.
6. **Automatic ROI masking** is well handled by local-variance / local-gradient-energy
   thresholding plus morphology and connected-component cleanup. There is no mature, validated,
   CPU-friendly segmentation model that beats this simple approach for laboratory speckle images,
   and ML masking would add an unverifiable dependency.
7. **Machine-learning DIC (Deep DIC / StrainNet / CNN-based) is promising but not yet a clearly
   superior, mature CPU tool.** Published studies show accuracy comparable to classic DIC on
   smooth fields and better on high-frequency fields, but they require GPU training, large
   synthetic datasets, and — importantly — several report that deep networks can perform *worse*
   than classic DIC on simple cases such as rigid body motion and can inherit pattern-induced
   bias. The prompt explicitly restricts us to CPU and mature tools, so ML is surveyed but not
   adopted.
8. **Uncertainty must be reported two ways:** (a) a per-point, model-based estimate from the
   matching residual propagated through the correlation Hessian (`Var(Δp) ≈ σ² (JᵀJ)⁻¹`,
   Wang et al. 2015), and (b) an independent, synthetic ground-truth validation with a
   zero-motion noise floor (iDICs GPG; DIC Challenge 1.0/2.0).
9. **Fundamental 2D-DIC limitations** that must be surfaced to the user: out-of-plane motion
   causes a scale (magnification) error proportional to the out-of-plane displacement divided by
   the camera-to-object distance; lens distortion and a non-parallel camera/specimen axis cause
   spatially varying bias; the small-strain definition is invalid above a few percent strain; and
   strain resolution is limited by the virtual strain gauge, not by the pixel grid.

---

## 2. Candidate approaches

### 2.1 Local (subset-based) DIC
Divide the ROI into overlapping square subsets; for each subset, find the warp that best matches
the reference subset in the deformed image.

* **Pros.** Embarrassingly parallel; robust to locally bad data; no global solve; the state of the
  art for accuracy and speed; simple to mask (skip subsets whose reference footprint leaves the
  ROI). This is what Ncorr, DICe, OpenCorr and py2DIC implement.
* **Cons.** The recovered field is a low-pass-filtered version of the true field (each subset
  reports an averaged, affine fit); near a mask edge the subset footprint must stay inside the
  specimen, so data are lost around the boundary; strain needs a separate fitting step.
* **Accuracy evidence.** Sub-pixel noise floors ~0.01–0.05 px; displacement bias can reach
  ~0.004 px with the best interpolants (Zhao 2019 review; Pan & Wang 2016).

### 2.2 Global / finite-element DIC
Discretise the displacement field over the whole ROI with FE shape functions and solve one large
non-linear least-squares problem.

* **Pros.** Kinematic compatibility is enforced by construction; natural coupling to FE models;
  excellent for heterogeneous fields and for high spatial resolution at low noise (Hild & Roux;
  μDIC; the new convolution-FE variant of Lu & Zhu 2024 outperformed classical FE-DIC on the
  DIC Challenge 2.0 benchmarks).
* **Cons.** Requires a mesh and a global linear solve each iteration; harder to make robust and
  fast for arbitrary input with no FE model available; boundary/masking bookkeeping is more
  complex; for a general-purpose CLI tool that must "just work" on an unknown specimen, the
  subset method is more forgiving.
* **Verdict.** Rejected as the *primary* engine for this task (no FE model is supplied, and the
  requirements stress generality and robustness), but noted as the natural next step for a user
  who wants FE-consistent fields.

### 2.3 Frequency-domain / global spectral DIC (LSA, Fourier DIC)
Formulate the optical residual in the Fourier domain.

* **Pros.** Very fast; optimal noise propagation; well suited to dense, pixel-wise fields
  (Blaysat, Grédiac & Sur).
* **Cons.** Assumes a stationary, periodic pattern and a global (often translational) model;
  struggles with arbitrary masks and localized phenomena; less intuitive to validate against a
  point-wise ground truth.
* **Verdict.** Rejected for this task.

### 2.4 Learning-based DIC (Deep DIC, StrainNet, CNN optical flow)
Train a CNN to map an image pair to a displacement (and sometimes strain) field.

* **Pros.** Fast inference after training; can be pixel-dense; competitive accuracy on
  high-spatial-frequency fields; robust to some pattern degradation (Deep DIC, Yang et al. 2022).
* **Cons.** Needs large, realistic synthetic training sets and GPU time; several studies report
  no advantage or a *disadvantage* on simple cases (rigid body motion, stationary images), and
  deep networks can absorb and reproduce pattern-induced bias (Boukhtache et al. 2020/2021);
  the trained model is a large, opaque dependency. The prompt scopes out training ML models and
  GPU work.
* **Verdict.** Surveyed, **not adopted**. It does not meet the "mature, validated, clearly
  superior, CPU" bar.

### 2.5 Comparison table

| Approach | Accuracy (reported) | Robustness | Cost (CPU) | Maturity | Complexity to build | Verdict |
|---|---|---|---|---|---|---|
| **Local subset + IC-GN, ZNSSD, affine, B-spline** | 0.01–0.05 px noise floor; bias ~0.004–0.01 px | High; per-point failure isolated; easy masking | Low–moderate; vectorisable | Very high (Ncorr, DICe, OpenCorr, commercial) | Moderate | **Chosen** |
| Global / FE-DIC | Excellent on smooth/heterogeneous fields; needs mesh | High if regularised; mesh-dependent | Moderate–high (global solve) | High (μDIC, commercial) | High | Rejected (no FE model; less general) |
| Fourier / LSA | Very good, near-optimal noise propagation | Moderate (needs stationary pattern) | Very low | Medium | Moderate | Rejected (masks, generality) |
| Deep learning (Deep DIC / StrainNet) | Comparable to DIC on smooth fields, better on high-frequency; can be worse on trivial cases | Pattern-dependent; opaque | Inference low, training high (GPU) | Emerging | High (+ GPU, data) | Rejected (out of scope, not mature) |

---

## 3. Best practice within the chosen family

### 3.1 Correlation criterion
**ZNSSD.** It removes the affine illumination nuisance (offset and gain) that corrupts SSD and
NSSD, and is the criterion for which the IC-GN Hessian is naturally formulated. Formally, for a
reference subset `f` and deformed subset `g` sampled through the current warp `p`,

```
C_ZNSSD = Σ_ξ [ (f(ξ) − f̄)/Δf − (g(W(ξ;p)) − ḡ)/Δg ]²,
Δf = sqrt(Σ (f − f̄)²),   Δg = sqrt(Σ (g − ḡ)²).
```

**Ncorr** uses ZNSSD; **OpenCorr**, **DICe** and the DIC-Challenge reference analyses all use
ZNSSD with affine shape functions and bicubic B-spline interpolation.

### 3.2 Shape / kinematic model
**Affine (first-order): 6 parameters** `p = (u, v, u_x, u_y, v_x, v_y)`. It captures translation,
rotation, normal and shear strain inside the subset. Second-order shape functions can lower bias
for strongly non-uniform fields, but they double the parameter count, slow convergence and are
unnecessary for the affine ground truth used here. The DIC Challenge analyses and iDICs GPG
treat affine + B-spline as the reference configuration.

### 3.3 Optimisation
**Inverse-compositional Gauss–Newton.** The reference-side quantities
`SD_i = ∇f · ∂W/∂p_i` (six *steepest-descent images*) and the constant Hessian
`H = Σ (SD SDᵀ)/Δf²` are precomputed once per point. Each iteration evaluates only the deformed
image through the current warp, forms the residual, and solves `Δp = −H⁻¹ Σ SD·r/Δf`, then
composes the inverse incremental warp into the current warp. Convergence is quadratic; ~4–8
iterations are typical. This is strictly more efficient than FA-NR at equal accuracy and is
recommended as the standard sub-pixel solver (Pan & Wang 2016).

### 3.4 Sub-pixel interpolation
**B-spline interpolation (order 3 or 5), prefiltered.** Bicubic *B-spline* has far less bias than
a bicubic *polynomial* (iDICs GPG 2025). The interpolant must also supply image gradients; the
B-spline derivative is obtained exactly by evaluating the order-`n−1` B-spline on the order-`n`
coefficients at half-pixel-shifted coordinates. A Gaussian pre-filter of the images reduces
interpolation bias and variance at the cost of a little spatial resolution, and iDICs treats
"whether you pre-filtered" as a reporting requirement.

### 3.5 Large deformation, rotation, incremental reference updating
* For a pure uniform stretch, a single reference works for the whole sequence as long as the
  pattern has not decorrelated; affine shape functions make the affine ground truth *exact*.
* When the pattern decorrelates (large rotation/stretch, illumination change), update the
  reference. **Do not** update on every frame: incremental accumulation of interpolation and
  systematic error is the dominant error source, and conditional updating (only when the
  correlation residual exceeds a threshold) minimises it (incremental-DIC error analysis 2023;
  feature-based incremental DIC 2023).
* Use a **temporal initial guess** (extrapolate the previous frames' warp) to keep the IC-GN
  basin small, and fall back to a **coarse multi-scale search** when the guess is poor. This is
  the mechanism that gives robustness to large inter-frame motion without paying the incremental
  accumulation penalty.

### 3.6 Strain from a displacement field
Use a **local first-order polynomial (plane) fit** over a strain window of displacement samples —
the *virtual strain gauge*. The fit's linear coefficients are the displacement gradients. The
effective gauge length is `L_VSG = (window − 1)·step + subset` (iDICs GPG Ch. 5), which is the
length over which strain is spatially averaged. Larger windows reduce strain noise but blur
strain gradients: this is the fundamental spatial-resolution/noise trade-off (DIC Challenge 2.0;
Rossi et al. 2015; the "9×9 strain window / 60 px subset" recipe of the small-strain study).
Alternative estimators (subset shape-function coefficients, FE shape functions, spline fit) exist;
the plane fit is the most transparent and is what DICe's virtual strain gauge does.

### 3.7 Masking non-specimen regions
* **Manual** masks (the `Data01/masks/` PNGs) are the ground truth for the user-mask path.
* **Automatic**: speckle regions have high local grey-level variance / gradient energy; empty
  background is flat. Compute a local standard-deviation (or Laplacian-energy) map, threshold it
  (Otsu, or a robust fraction of the map), then clean up with morphological opening/closing,
  hole filling and connected-component filtering. Finally **erode by the subset half-width** so
  that every retained subset footprint lies entirely inside the specimen.
* **Modern segmentation** (semantic CNNs) can mask specimens in photographs, but requires
  training data and a GPU, and offers no accuracy guarantee for the ROI that actually matters
  here. Not adopted.

### 3.8 Uncertainty and accuracy estimation
Two complementary routes, both standard:
* **Model-based, per point:** the covariance of the estimated parameters is
  `Var(Δp_j) ≈ σ² (JᵀJ)⁻¹_jj` where `J` is the correlation Jacobian and `σ²` the residual noise
  variance (Wang et al. 2015; Sutton et al. 2009). Random displacement error is therefore
  proportional to image noise and inversely proportional to the sum of squared intensity
  gradients (SSSIG) in the subset. This yields a per-point σ_u, σ_v.
* **Empirical:** the **noise floor** — the apparent motion produced on data with zero true
  motion — and comparison against **synthetic ground truth** with known fields. iDICs recommends
  a noise-floor analysis as part of any DIC measurement report.

### 3.9 Reference open-source packages

| Package | Language / license | Notes / lessons |
|---|---|---|
| **Ncorr** | MATLAB + C++/MEX, **BSD** | Reference 2D subset DIC; RG-DIC seeding; circular subsets; documented reliability-guided propagation and DIC-Challenge images. Its own disclaimer stresses that the code cannot *guarantee* correctness — validation is on the user. |
| **DICe** | C++, Sandia, **BSD** | Virtual strain gauge, flexible shape functions, strong on uncertainty quantification; used in DIC Challenge comparisons. |
| **OpenCorr** | C++, **open source (CC-BY article; library on GitHub)** | Developer-oriented library with 2D/stereo/DVC, IC-GN, feature-guided DIC handling >35 % strain, self-adaptive subsets. |
| **py2DIC** | Python, **GPL-family FOSS** | Simple template-matching Python DIC; benchmarked against Ncorr/VIC-2D/DICe on DIC-Challenge Sample 12; a useful demonstration that a modest Python implementation agrees with mature codes. |
| **Pyvale** | Python/C++ (2025 preprint) | Modern, scalable 2D DIC engine; uses 31 px subsets, step 15, ZNSSD, affine shape functions, bicubic B-spline — the same configuration chosen here. |
| **MatchID / DaVis / VIC-2D** | commercial | Reference features/conventions; closed source. |

**License note.** We implement the core ourselves from the literature. No code is copied from
these projects. BSD-licensed Ncorr/DICe would in principle permit reuse, but a clean,
well-understood implementation is easier to validate and document than an integration.

### 3.10 Fundamental limitations of 2D DIC (must be told to the user)
1. **Out-of-plane motion.** 2D DIC assumes a planar specimen parallel to the sensor. A
   through-thickness displacement `w` changes the apparent scale by ≈ `w/Z` (Z = camera distance),
   producing a spurious strain; a 1 mm out-of-plane motion at 1 m is ~0.1 % strain.
2. **Lens distortion / non-parallel axes.** Radial distortion and a tilted camera give a
   spatially varying bias that the calibration-free 2D pipeline cannot remove.
3. **Small-strain validity.** The engineering (small-strain) definition `ε = (l−l₀)/l₀` and the
   linearised `∂u/∂x` differ from finite strain by O(ε²); above a few percent the difference is
   significant. A finite-strain (Green–Lagrange) measure must be offered.
4. **Speckle quality.** Speckle diameter should be ~3–5 px and coverage ~50 %; too-small speckles
   alias, too-large speckles force big subsets and poor spatial resolution.
5. **Spatial resolution vs noise.** Strain is a derivative: it amplifies displacement noise, and
   the only cure (a larger virtual strain gauge) blurs real gradients.
6. **Pattern-induced bias (PIB).** A periodic or overly regular pattern, or a very smooth one,
   biases the result; DIC Challenge 2.0 quantifies it.
7. **Noise floor is spatial.** Uncertainty varies across the field with local contrast
   (Peshave et al. 2025), so a single global number is insufficient — per-point uncertainty is
   needed.

---

## 4. Recommendation

**Implement local subset-based 2D DIC with:**
* ZNSSD matching criterion,
* affine (first-order, 6-parameter) shape functions,
* IC-GN optimisation,
* prefiltered B-spline interpolation (order 3, optional 5) with exact B-spline gradients,
* direct correlation against the original reference by default, with a temporal initial guess and
  a coarse multi-scale affine fallback, plus an *optional* incremental reference-updating mode,
* automatic ROI masking by local-variance thresholding + morphology + erosion, with a user-mask
  override,
* strain from a local plane fit (virtual strain gauge) with engineering **and** Green–Lagrange
  measures, plus rotation, principal and equivalent strains,
* per-point uncertainty from the correlation Hessian and residual, and a synthetic zero-motion
  noise floor,
* a full synthetic validation suite (`--validate`) covering all the required cases.

**Why this fits §4 of the prompt:** it is a CPU tool, deterministic, fully explainable, easy to
mask, robust per point, and it reaches the published accuracy floor (0.01–0.05 px) on
well-speckled data. It handles the provided large-stretch dataset exactly (affine shape functions
are exact for affine deformation) and, with the coarse fallback and optional incremental mode,
handles rotation and large translation too.

**Why not the alternatives:** global/FE DIC needs a mesh and a global solver and is less
forgiving on unknown inputs; Fourier/LSA needs a stationary pattern and struggles with arbitrary
masks; ML-DIC is out of scope (no training, no GPU), not clearly superior on smooth fields, and
opaque for a validation report.

---

## 5. Assumptions and limitations the user must know

* **Planar specimen, parallel to the sensor.** Out-of-plane motion and camera tilt are not
  corrected. See §3.10.
* **One camera, no calibration.** Only in-plane displacement is measured; the physical scale
  (`--scale`, mm/px) is user-supplied and constant across the field.
* **Constant illumination model.** ZNSSD is invariant to a global offset and gain, but not to a
  spatially varying or non-linear illumination change (vignetting, shadows, specular hotspots).
* **Speckle-like texture assumed** in the ROI. The automatic masker detects "textured vs flat";
  a flat but textured background (e.g. a printed grid) would be kept.
* **Strain validity.** Engineering strain is reported but flagged invalid above a few percent;
  Green–Lagrange strain is the finite-strain measure to trust at large deformation.
* **Spatial resolution is the virtual strain gauge, not the pixel.** The reported strain is an
  average over `L_VSG`; peaks narrower than that are attenuated.
* **Displacement is defined at the subset centre** and is a subset-averaged affine fit; the
  field is a low-pass version of the truth.
* **Reference-frame results** are relative to the first image unless `--reference incremental`
  is used; incremental mode trades accumulation for decorrelation robustness.

---

## 6. Accuracy targets derived from the literature

* Displacement noise floor (zero motion, σ_grey ≈ 2, well-speckled): **≤ 0.05 px RMS**, target
  **≤ 0.02 px**.
* Displacement bias on a synthetic rigid translation: **≤ 0.02 px** mean.
* Displacement RMS error vs analytic ground truth (uniform stretch): **≤ 0.05 px**.
* Strain noise floor: **≤ 5×10⁻⁴** (0.05 %) for a ≥ 5×5 strain window.
* Strain error vs analytic ground truth: **≤ 2×10⁻³** (0.2 %) absolute.
* Coverage: ≥ 90 % of ROI grid points valid on the provided dataset where ground truth is valid.
* Speed: ≤ ~60 s/frame for ~3–4 k points on 1024² images on a modern CPU.

These are the targets recorded in `PLAN.md` and enforced by `--validate`.

---

## 7. Open questions / risks

* **Very large inter-frame motion** (single-step > the search radius). Mitigated by the coarse
  multi-scale initial guess; documented as a user-tunable parameter.
* **Illumination change beyond affine.** ZNSSD tolerates offset/gain only; a linear illumination
  gradient is partially handled by the affine subset model, but strong vignetting is not.
* **Very smooth / low-contrast speckle** lowers the SSSIG and raises the noise floor; the tool
  reports per-point uncertainty so the user can see where it degrades.

---

## 8. Impact of the research on the design (traceability)

| Research finding | Design consequence |
|---|---|
| IC-GN beats FA-NR on speed and noise robustness | IC-GN core with precomputed Hessian/steepest descent |
| ZNSSD is illumination-invariant and standard | ZNSSD criterion; report normalised residual |
| Affine + B-spline is the reference configuration | affine shape functions; prefiltered B-spline, exact gradients |
| Incremental updating accumulates error | direct-reference default; incremental is opt-in |
| Strain = virtual strain gauge with a resolution/noise trade-off | plane-fit strain window; report `L_VSG` |
| Uncertainty `≈ σ² (JᵀJ)⁻¹`; noise floor is spatial | per-point σ_u, σ_v maps + zero-motion noise floor |
| Automatic masking by local variance is adequate | variance/Laplacian-energy masker + morphology + erosion |
| ML-DIC not mature/superior on CPU | not adopted (documented) |
| 2D-DIC fundamental limits | listed in README/limitations and the summary report |

---

## 9. Sources consulted (verified)

All links were opened during the survey (HTTP 200; the few `403`/`429` responses are
publisher bot-blocks on the same canonical URLs, whose titles and abstracts were retrieved via
search). Titles are as published.

1. **International Digital Image Correlation Society, E. M. C. Jones & M. A. Iadicola (Eds.),
   *A Good Practices Guide for Digital Image Correlation*, 2018, Ed. 2 (2025).**
   <https://idics.org/guide/GPG_Ed2_251104.pdf> — subset size/step rules of thumb, subset shape
   function, interpolant, virtual strain gauge `L_VSG`, displacement noise floor, uncertainty
   categories (variance vs bias), reporting requirements.
2. **B. Pan, B. Wang, "Digital Image Correlation with Enhanced Accuracy and Efficiency: A
   Comparison of Two Subpixel Registration Algorithms", *Experimental Mechanics* 56, 1395–1409,
   2016.** <https://link.springer.com/article/10.1007/s11340-016-0180-z> — IC-GN vs FA-NR;
   IC-GN recommended; analytic bias/variance for common interpolants.
3. **B. Pan, K. Li, W. Tong, "Fast, Robust and Accurate Digital Image Correlation Calculation
   Without Redundant Computations", *Experimental Mechanics* 53, 1277–1289, 2013.**
   <https://doi.org/10.1007/s11340-013-9717-6> — the IC-GN algorithm as used here (described
   also in [2], [5], [8]).
4. **Y. Wang, P. Lava, R. Reu, P. Debruyne, et al., "Theoretical analysis on the random error of
   DIC", Sandia report / *Optics and Lasers in Engineering* 2015** — `Var(Δp_j) ≈ 2σ²(H⁻¹)_jj`;
   random error ∝ image noise, ∝ 1/SSSIG.
   <https://www.osti.gov/servlets/purl/1319535>
5. **B. Pan et al., "Noise robustness and parallel computation of the inverse compositional
   Gauss–Newton algorithm in DIC", *Optics and Lasers in Engineering*, 2015.**
   <https://www.sciencedirect.com/science/article/pii/S0143816615000445>
6. **Y. Zhao, "The state of the art of two-dimensional digital image correlation computational
   method", *Engineering Reports* 1(2), e12038, 2019.** — displacement resolution ~0.1–0.01 px,
   bias to 0.004 px; review of criteria and algorithms.
   <https://onlinelibrary.wiley.com/doi/full/10.1002/eng2.12038>
7. **B. Pan, "Digital image correlation for surface deformation measurement: historical
   developments, recent advances and future goals", *Meas. Sci. Technol.* 29, 082001, 2018.**
   <https://iopscience.iop.org/article/10.1088/1361-6501/aac55b>
8. **Z. Jiang, "OpenCorr: An open source library for research and development of digital image
   correlation", *Optics and Lasers in Engineering* 169, 107566, 2023.**
   <https://www.sciencedirect.com/science/article/pii/S0143816623000957> — feature-guided DIC
   handling >35 % strain; self-adaptive subsets.
9. **J. Blaber, B. Adair, A. Antoniou, "Ncorr: Open-Source 2D Digital Image Correlation Matlab
   Software", *Experimental Mechanics* 55, 1105–1122, 2015.** <http://www.ncorr.com/> —
   BSD-licensed reference implementation; RG-DIC seeding; DIC-Challenge sample images.
10. **V. Belloni et al., "py2DIC: A New Free and Open Source Software for Displacement and Strain
    Measurements in the Field of Experimental Mechanics", *Sensors* 19(18), 3832, 2019.**
    <https://www.mdpi.com/1424-8220/19/18/3832> — Python DIC compared with Ncorr, VIC-2D, DICe
    on DIC-Challenge Sample 12.
11. **P. L. Reu et al., "DIC Challenge: Developing Images and Guidelines for Evaluating Accuracy
    and Resolution of 2D Analyses", *Experimental Mechanics* 58, 1067–1099, 2018.**
    <https://www.nist.gov/publications/dic-challenge-20-developing-images-and-guidelines-evaluating-accuracy-and-resolution-2d>
    and **"DIC Challenge 2.0 ... Metrological Efficiency Indicator", *Exp. Mech.*, 2022** —
    spatial-resolution definition, displacement/strain noise vs resolution trade-off, PIB.
12. **A. Peshave, F. Pierron, P. Lava, et al., "Practical Uncertainty Quantification Guidelines for
    DIC-Based Numerical Model Validation", *Experimental Techniques* 49, 437–457, 2025.**
    <https://link.springer.com/article/10.1007/s40799-024-00758-1> — spatially varying noise
    floors; reference-image noise causes temporal bias; deformed-image noise causes temporal
    scatter.
13. **S. Boukhtache, K. Abdelouahab, F. Berry, B. Blaysat, M. Grédiac, F. Sur, "When Deep
    Learning Meets Digital Image Correlation", *Optics and Lasers in Engineering*, 2021.**
    <https://arxiv.org/abs/2009.03993> — CNN (StrainNet) comparable to classic DIC at low
    frequency, better at high frequency; limited generalisation; PIB.
14. **R. Yang, Y. Li, D. Zeng, P. Guo, "Deep DIC: Deep Learning-Based Digital Image Correlation
    for End-to-End Displacement and Strain Measurement", *J. Mater. Process. Technol.*, 2022.**
    <https://arxiv.org/abs/2110.13720> — deep DIC can under-perform classic DIC on trivial cases
    (stationary, rigid motion).
15. **Y. Lu, W. Zhu, "Convolution finite element-based digital image correlation for displacement
    and strain measurements", *CMAME* 419, 116597, 2024.** <https://arxiv.org/abs/2311.03504> —
    global FE-DIC context; outperformed classical FE-DIC on DIC Challenge 2.0.
16. **J.-C. Passieux et al., "Classic and inverse compositional Gauss-Newton in global DIC",
    *Int. J. Numer. Meth. Engng.*, 2019.** <https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.6057>
17. **"Error analysis and practical considerations of incremental digital image correlation",
    *Chinese J. Applied Mechanics*, 2023.** — conditional reference updating minimises accumulated
    error. <https://lxsj.cstam.org.cn/en/article/doi/10.6052/1000-0879-23-275>
18. **"A novel feature-based incremental digital image correlation method to measure large
    deformations of soft materials", 2023.** — incremental accumulation; threshold-based reference
    selection.
    <https://www.bssm.org/media/p3fbr0yq/a-novel-feature-based-incremental-digital-image-correlation-method-to-measure-large-deformations-of-soft-materials.pdf>
19. **D. Z. Turner, "An overview of the virtual strain gauge formulation in DICe", Sandia report
    SAND2018.** <https://www.osti.gov/servlets/purl/1528762> — plane-fit strain post-processing.
20. **Y. Wang et al., "Experimentally Achievable Accuracy Using a DIC Technique in measuring
    Small-Magnitude (<0.1%) Homogeneous Strain Fields", *Sensors*, 2018.**
    <https://pmc.ncbi.nlm.nih.gov/articles/PMC5978128> — subset size/step/strain-window recipe;
    strain noise decreases with larger strain window.
21. **M. Rossi, P. Lava, F. Pierron, D. Debruyne, M. Sasso, "Effect of DIC Spatial Resolution,
    Noise and Interpolation Error on Identification Results with the VFM", *Strain* 51, 206–222,
    2015.** — resolution/noise trade-off.
22. **Pyvale (2025 preprint), "A Fast, Scalable, Open-Source 2D DIC Engine Capable of Handling
    Gigapixel Images".** <https://arxiv.org/html/2601.12941v2> — modern configuration
    (31 px subset, step 15, ZNSSD, affine, bicubic B-spline); runtime comparison of Ncorr,
    OpenCorr, DICe.
23. **iDICs DIC-101 course notes, Ch. 4–5 (Sandia), 2020.** <https://www.idics.org/courses/dic101/>
    — correlation examples, pre-filtering effect, matching criteria, subset/step selection.

*(Sutton, Orteu & Schreier, "Image Correlation for Shape, Motion and Deformation Measurements",
Springer 2009 — the standard textbook — is used for terminology and theory throughout.)*
