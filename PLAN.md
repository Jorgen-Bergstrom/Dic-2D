# PLAN.md — milestones, accuracy targets, and expected trade-offs

Derived from the literature surveyed in `docs/RESEARCH.md`. This is a living document; the
final, *measured* numbers live in `validation_report.md`.

---

## 1. Milestones

| # | Milestone | Deliverable | Gate (must pass before moving on) |
|---|---|---|---|
| M0 | Research survey | `docs/RESEARCH.md` | Done: verified sources, defended recommendation |
| M1 | Skeleton & packaging | `pyproject.toml`, `src/dic/*`, `python -m dic --help` | Package installs; `--help` lists every option/default/unit |
| M2 | Interpolation + kinematic warp | `interpolation.py`, `kinematics.py` | Unit test: B-spline interpolation and its exact derivative reproduce an analytic function; warp round-trips |
| M3 | Correlation core (IC-GN) | `correlation.py` | Unit test: recover a known sub-pixel translation to <0.02 px on a synthetic patch; recover a known affine warp |
| M4 | ROI / masking | `mask.py` | Unit test: auto-mask finds a synthetic specimen and ignores flat background; user mask honoured |
| M5 | Strain estimation | `strain.py` | Unit test: plane fit recovers analytic gradients of an affine/quadratic displacement field to <1e-6 |
| M6 | Pipeline end-to-end | `pipeline.py`, `io.py`, `cli.py` | Smoke test on 3 small synthetic frames |
| M7 | Uncertainty | `uncertainty.py` | Per-point σ and zero-motion noise floor produced and consistent with theory |
| M8 | Visualization | `visualization.py` | Per-frame displacement + strain overlays, animations, curves, profiles |
| M9 | Validation suite | `validation.py`, `synth.py`, `--validate` | All §5.5b synthetic cases pass their pre-stated criteria |
| M10 | Validation on `Data01/` | `validation_report.md` §A | Displacement/strain errors vs `ground_truth_grid.csv` meet targets |
| M11 | Docs & demo | `README.md`, `DECISIONS.md`, `docs/ALGORITHM.md`, `examples/` | One-command demo runs; README covers install/quick-start/CLI/conventions/limits |

Rule of engagement: **never move past a milestone with failing tests.** `pytest -q` must be green
at every gate.

---

## 2. Quantitative accuracy targets

Justification in brackets. Targets are checked by `--validate` and by the `Data01/` comparison.

### 2.1 Displacement
| Metric | Target | Basis |
|---|---|---|
| Zero-motion noise floor, σ_grey ≈ 2 | **≤ 0.05 px RMS**, aim **≤ 0.02 px** | iDICs noise-floor ~0.01 px; Zhao 2019 reports 0.01–0.1 px |
| Sub-pixel rigid translation bias | **≤ 0.02 px** | Pan & Wang 2016 bias ≈ 0.004–0.01 px |
| Multi-pixel translation RMS error | **≤ 0.03 px** | same |
| Uniform stretch RMS error (Data01, comparable points) | **≤ 0.05 px** | affine truth ⇒ interpolation+noise limited |
| Rigid rotation (≥30°) RMS error | **≤ 0.05 px** | affine shape functions are exact for rotation |
| Affine shear / non-uniform field RMS error | **≤ 0.05 px** | |

### 2.2 Strain
| Metric | Target | Basis |
|---|---|---|
| Zero-motion strain noise floor (≥5×5 window) | **≤ 5×10⁻⁴** | strain-noise studies; window smoothing |
| Uniform stretch strain error | **≤ 2×10⁻³** absolute | DIC-Challenge-class agreement |
| Analytic-gradient (non-uniform) strain error | **≤ 2×10⁻³** absolute | plane fit is exact for linear fields |
| Rotation produces ~zero strain | **≤ 1×10⁻³** residual | |

### 2.3 Coverage, robustness, speed
| Metric | Target |
|---|---|
| Coverage on `Data01/` where ground truth valid | ≥ 90 % |
| Points written as `NaN` when non-converged/out-of-frame | 100 % (never fabricated) |
| Throughput, 1024², ~3–4 k points, CPU | ≤ ~60 s/frame |
| Reproducibility | byte-identical results for identical inputs/params/seed |

---

## 3. Expected accuracy-vs-cost trade-offs (needed for §5.6 reporting)

These are the qualitative laws the report will quantify with a parameter sweep.

### 3.1 Speckle quality
* Speckle diameter ~3–5 px, coverage ~50 % is optimal (iDICs; Reu speckle studies).
* Too small ⇒ aliasing ⇒ higher noise and bias. Too large ⇒ the subset must be large to contain
  enough features ⇒ worse spatial resolution.
* Noise floor scales as `σ_u ∝ σ_grey / sqrt(Σ (∂I/∂x)²)`: sharp, high-contrast speckle (large
  SSSIG) lowers the noise floor. **Effect: roughly 1/contrast.**

### 3.2 Image noise
* Random displacement error ∝ image noise σ_grey (Wang et al. 2015). **Doubling noise doubles the
  displacement noise floor.**
* Strain noise is amplified by the derivative: it scales like `σ_ε ∝ σ_u / L_VSG`.

### 3.3 Subset size
* Larger subset ⇒ larger SSSIG ⇒ lower noise floor (∝ 1/sqrt(subset area) for fixed speckle),
  but ⇒ stronger low-pass filtering and more data lost near mask edges.
* Rule of thumb: subset must contain ≥ 3 speckles of 3–5 px.
* **Effect: noise floor roughly ∝ 1/subset size; spatial resolution roughly ∝ subset size.**

### 3.4 Grid step
* Smaller step ⇒ denser sampling (better peak capture) but the samples are correlated, so the
  *independent* spatial resolution is set by the subset/strain window, not the step. Step ≈
  1/3–1/2 of subset is recommended.

### 3.5 Strain window
* Strain noise ∝ 1/(window) for a plane fit; spatial resolution (gauge length)
  `L_VSG = (window−1)·step + subset`.
* **Effect: doubling the window roughly halves the strain noise and doubles the gauge length.**
* Recommendation to the user: choose the smallest window that keeps the strain noise floor below
  the strain change they need to resolve.

### 3.6 Interpolation order
* Bicubic (order 3) B-spline is the standard; biquintic (order 5) lowers interpolation bias at
  ~2× the interpolation cost. Expected improvement at sub-0.01 px level only.

---

## 4. Parameter sweep plan (for §5.6)

Run a controlled sweep on synthetic ground truth (zero motion and uniform stretch) and record, for
the final report:
* subset ∈ {21, 31, 41, 61} at fixed step;
* strain window ∈ {3, 5, 7, 9};
* noise σ ∈ {0, 1, 2, 5};
* interpolant order ∈ {3, 5}.
Report displacement RMS and strain RMS as functions of each, and derive the user-facing
"expected error for your settings" table.

---

## 5. Risk register

| Risk | Mitigation |
|---|---|
| Pure-Python IC-GN too slow | Vectorise over points; precompute Hessian/steepest descent; optional `numba` guarded by fallback |
| Large inter-frame motion exceeds search radius | Temporal initial guess + coarse multi-scale affine fallback; `--search`/`--init` options |
| Out-of-frame points fabricated | Bounds check every target sample; invalid ⇒ `NaN`; coverage reported |
| Overstated accuracy | Independent synthetic validation with pre-stated criteria; noise floor measured, not assumed |
| Non-convergence silently accepted | ZNCC/SSSIG thresholds; failed points invalid and counted |
| `Data01/` modified | Opened read-only; every output written to a separate directory |
