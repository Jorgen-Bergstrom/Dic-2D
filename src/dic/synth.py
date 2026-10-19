"""Analytic synthetic speckle generator with exact ground truth.

Images are rendered by *stamping* each speckle analytically (no interpolation of a raster), so a
deformed image is the exact push-forward of the reference pattern.  That makes the displacement
and strain ground truth exact, which is what the validation suite needs.

A pattern is a set of circular speckles with a tapered (soft-edged) profile on a uniform
background.  Under a deformation ``x = phi(X)`` each speckle centre moves to ``phi(X_c)`` and its
shape is warped by the local Jacobian ``J = d phi / d X``, so ellipses/shears are rendered
correctly for affine and slowly varying fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------------------
# Deformations (reference -> deformed), with analytic Jacobians
# --------------------------------------------------------------------------------------
class Deformation:
    """Base class: a smooth map ``X -> x`` with an analytic Jacobian."""

    def __call__(self, pts: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def jacobian(self, pts: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def description(self) -> str:
        return self.__class__.__name__


@dataclass
class AffineDeformation(Deformation):
    """``x = F (X - c) + c + t`` (F = deformation gradient, c = pivot, t = translation)."""

    F: np.ndarray
    t: np.ndarray = None  # type: ignore[assignment]
    center: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.F = np.asarray(self.F, dtype=np.float64)
        self.t = np.zeros(2) if self.t is None else np.asarray(self.t, dtype=np.float64)
        self.center = (
            np.zeros(2) if self.center is None else np.asarray(self.center, dtype=np.float64)
        )

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        out = (p - self.center) @ self.F.T + self.center + self.t
        return out if np.ndim(pts) == 2 else out[0]

    def jacobian(self, pts: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        n = p.shape[0]
        J = np.broadcast_to(self.F, (n, 2, 2)).copy()
        return J if np.ndim(pts) == 2 else J[0]

    @property
    def description(self) -> str:
        return f"Affine F={self.F.tolist()} t={self.t.tolist()}"


@dataclass
class NonUniformDeformation(Deformation):
    """A smooth analytic field with controllable gradients.

    ``u_x = a * sin(k_y * Y) + s_xx * (X - cx) + s_xy * (Y - cy)``
    ``u_y = b * sin(k_x * X) + s_yy * (Y - cy) + s_yx * (X - cx)``

    Parameters
    ----------
    a, b : float
        Amplitude (px) of the sinusoidal part.
    kx, ky : float
        Angular wavenumbers (rad/px).
    shear : tuple of 4 floats
        ``(s_xx, s_xy, s_yx, s_yy)`` linear part of the displacement gradient.
    center : array_like
        Pivot for the linear part.
    """

    a: float = 3.0
    b: float = 2.0
    kx: float = 2.0 * np.pi / 300.0
    ky: float = 2.0 * np.pi / 400.0
    shear: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    center: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.center = (
            np.zeros(2) if self.center is None else np.asarray(self.center, dtype=np.float64)
        )

    def _disp(self, pts: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(pts)
        X = p[:, 0] - self.center[0]
        Y = p[:, 1] - self.center[1]
        s_xx, s_xy, s_yx, s_yy = self.shear
        ux = self.a * np.sin(self.ky * Y) + s_xx * X + s_xy * Y
        uy = self.b * np.sin(self.kx * X) + s_yy * Y + s_yx * X
        return np.stack([ux, uy], axis=1)

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        out = p + self._disp(p)
        return out if np.ndim(pts) == 2 else out[0]

    def jacobian(self, pts: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        X = p[:, 0] - self.center[0]
        Y = p[:, 1] - self.center[1]
        s_xx, s_xy, s_yx, s_yy = self.shear
        n = p.shape[0]
        J = np.empty((n, 2, 2))
        # x_out = X + ux  -> d x_out / dX
        J[:, 0, 0] = 1.0 + s_xx
        J[:, 0, 1] = self.a * self.ky * np.cos(self.ky * Y) + s_xy
        # y_out = Y + uy  -> d y_out / dX
        J[:, 1, 0] = self.b * self.kx * np.cos(self.kx * X) + s_yx
        J[:, 1, 1] = 1.0 + s_yy
        return J if np.ndim(pts) == 2 else J[0]

    @property
    def description(self) -> str:
        return (
            f"NonUniform a={self.a} b={self.b} kx={self.kx:.4f} ky={self.ky:.4f} "
            f"shear={self.shear}"
        )


def rotation(angle_rad: float, center: np.ndarray) -> AffineDeformation:
    """Rigid rotation about *center*."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return AffineDeformation(np.array([[c, -s], [s, c]]), np.zeros(2), center)


def stretch(ex: float, ey: float, center: np.ndarray) -> AffineDeformation:
    """Uniform normal stretch (engineering strains *ex*, *ey*)."""
    return AffineDeformation(np.diag([1.0 + ex, 1.0 + ey]), np.zeros(2), center)


def shear(gamma: float, center: np.ndarray) -> AffineDeformation:
    """Simple shear: engineering shear strain *gamma*."""
    return AffineDeformation(np.array([[1.0, gamma], [0.0, 1.0]]), np.zeros(2), center)


# --------------------------------------------------------------------------------------
# Speckle pattern
# --------------------------------------------------------------------------------------
@dataclass
class DotPattern:
    """A random speckle pattern of tapered circular dots."""

    x: np.ndarray
    y: np.ndarray
    radius: np.ndarray
    amplitude: np.ndarray
    softness: float = 0.7
    background: float = 128.0
    shape: tuple[int, int] = (512, 512)

    @classmethod
    def generate(
        cls,
        shape: tuple[int, int] = (512, 512),
        *,
        n_dots: int = 2500,
        radius_range: tuple[float, float] = (3.0, 6.0),
        amplitude_range: tuple[float, float] = (70.0, 120.0),
        background: float = 128.0,
        softness: float = 0.7,
        seed: int = 0,
        region: tuple[float, float, float, float] | None = None,
    ) -> DotPattern:
        """Create a random pattern covering *shape* (px).

        Parameters
        ----------
        region : tuple of float, optional
            ``(x0, y0, x1, y1)``; if given, dots are placed only inside this rectangle, leaving
            the rest of the frame as flat background (used for the auto-masking test).
        """
        rng = np.random.default_rng(seed)
        h, w = shape
        if region is None:
            x0, y0, x1, y1 = 0.0, 0.0, float(w), float(h)
        else:
            x0, y0, x1, y1 = region
        x = rng.uniform(x0, x1, n_dots)
        y = rng.uniform(y0, y1, n_dots)
        radius = rng.uniform(*radius_range, n_dots)
        amplitude = rng.uniform(*amplitude_range, n_dots)
        return cls(x, y, radius, amplitude, softness, background, shape)

    def n_dots_for_area(self, area_fraction: float = 0.5) -> int:
        """Not used directly; kept for API completeness."""
        h, w = self.shape
        mean_r = float(np.mean(self.radius))
        return int(area_fraction * h * w / (np.pi * mean_r ** 2))


def _profile(r: np.ndarray, radius: float, softness: float) -> np.ndarray:
    """Tapered disc profile in [0, 1]."""
    if softness <= 0:
        return (r <= radius).astype(np.float64)
    inner = radius - softness
    out = np.zeros_like(r)
    out[r <= inner] = 1.0
    band = (r > inner) & (r < radius)
    t = (r[band] - inner) / softness
    out[band] = 0.5 * (1.0 + np.cos(np.pi * t))
    return out


def render(
    pattern: DotPattern,
    deformation: Deformation | None = None,
    *,
    noise_sigma: float = 0.0,
    seed: int = 0,
    gain: float = 1.0,
    offset: float = 0.0,
    illum_gradient: tuple[float, float] = (0.0, 0.0),
    clip: bool = True,
) -> np.ndarray:
    """Render the pattern, optionally deformed, with noise and illumination change.

    Parameters
    ----------
    pattern : DotPattern
        The speckle pattern.
    deformation : Deformation, optional
        Reference-to-deformed map; ``None`` = identity (undeformed).
    noise_sigma : float
        Additive Gaussian noise standard deviation (grey levels).
    seed : int
        RNG seed for the noise.
    gain, offset : float
        Linear illumination change ``gain * I + offset``.
    illum_gradient : tuple of float
        Additional linear illumination ramp ``(dx, dy)`` in grey levels per pixel.
    clip : bool
        Clip to ``[0, 255]`` (mimics an 8-bit camera).

    Returns
    -------
    numpy.ndarray
        Float image, shape ``pattern.shape``.
    """
    h, w = pattern.shape
    img = np.full((h, w), pattern.background, dtype=np.float64)

    centers = np.stack([pattern.x, pattern.y], axis=1)
    if deformation is None:
        new_centers = centers
        jac = np.broadcast_to(np.eye(2), (centers.shape[0], 2, 2))
        max_sv = np.ones(centers.shape[0])
    else:
        new_centers = deformation(centers)
        jac = deformation.jacobian(centers)
        max_sv = np.linalg.norm(jac, axis=(1, 2))  # upper bound on stretch

    for k in range(centers.shape[0]):
        cx, cy = new_centers[k]
        R = pattern.radius[k]
        amp = pattern.amplitude[k]
        R_def = R * max_sv[k] + pattern.softness + 1.0
        x0 = max(0, int(np.floor(cx - R_def)))
        x1 = min(w, int(np.ceil(cx + R_def)) + 1)
        y0 = max(0, int(np.floor(cy - R_def)))
        y1 = min(h, int(np.ceil(cy + R_def)) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx = xx - cx
        dy = yy - cy
        try:
            Jinv = np.linalg.inv(jac[k])
        except np.linalg.LinAlgError:
            Jinv = np.eye(2)
        # reference-frame local coordinates
        rx = Jinv[0, 0] * dx + Jinv[0, 1] * dy
        ry = Jinv[1, 0] * dx + Jinv[1, 1] * dy
        r = np.sqrt(rx * rx + ry * ry)
        img[y0:y1, x0:x1] += amp * _profile(r, R, pattern.softness)

    if gain != 1.0 or offset != 0.0 or illum_gradient != (0.0, 0.0):
        yy, xx = np.mgrid[0:h, 0:w]
        img = gain * img + offset + illum_gradient[0] * xx + illum_gradient[1] * yy

    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        img = img + rng.normal(0.0, noise_sigma, size=img.shape)

    if clip:
        img = np.clip(img, 0.0, 255.0)
    return img


def ground_truth_fields(
    deformation: Deformation,
    xs: np.ndarray,
    ys: np.ndarray,
) -> dict[str, np.ndarray]:
    """Exact displacement and strain at reference coordinates ``(xs, ys)``.

    Returns
    -------
    dict with keys
        ``ux, uy`` (px), ``ux_x, ux_y, uy_x, uy_y`` (dimensionless gradients),
        ``exx, eyy, gamma_xy`` (engineering, image frame),
        ``Exx, Eyy, Exy`` (Green-Lagrange).
    """
    pts = np.stack([xs, ys], axis=1)
    moved = deformation(pts)
    J = deformation.jacobian(pts)
    u = moved - pts
    # displacement gradient H = J - I
    ux_x = J[:, 0, 0] - 1.0
    ux_y = J[:, 0, 1]
    uy_x = J[:, 1, 0]
    uy_y = J[:, 1, 1] - 1.0
    F = J  # deformation gradient (image frame)
    Exx = 0.5 * (F[:, 0, 0] ** 2 + F[:, 1, 0] ** 2 - 1.0)
    Eyy = 0.5 * (F[:, 0, 1] ** 2 + F[:, 1, 1] ** 2 - 1.0)
    Exy = 0.5 * (F[:, 0, 0] * F[:, 0, 1] + F[:, 1, 0] * F[:, 1, 1])
    return {
        "ux": u[:, 0],
        "uy": u[:, 1],
        "ux_x": ux_x,
        "ux_y": ux_y,
        "uy_x": uy_x,
        "uy_y": uy_y,
        "exx": ux_x,
        "eyy": uy_y,
        "gamma_xy": ux_y + uy_x,
        "Exx": Exx,
        "Eyy": Eyy,
        "Exy": Exy,
    }
