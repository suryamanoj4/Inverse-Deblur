"""PSF library and the versioned protocol kernel sets.

All kernels are returned normalised to unit mass. Protocol v1 (the
mid-evaluation protocol, frozen) uses isotropic Gaussian, linear motion and
defocus kernels. Protocol v2 adds anisotropic Gaussian, random-trajectory
camera shake (Boracchi & Foi, 2012), Airy and Moffat profiles, and optional
real recorded kernels (Levin et al., 2009; Sun et al., 2013; Kohler et al.,
2012) loaded from disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # Pillow is a core dependency, but keep the import local to I/O paths
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover
    Image = None


@dataclass(frozen=True)
class PSF:
    """A named point-spread function."""

    name: str
    kernel: np.ndarray
    kind: str

    def __post_init__(self) -> None:
        k = np.asarray(self.kernel, dtype=np.float64)
        if k.ndim != 2:
            raise ValueError(f"PSF kernel must be 2D, got shape {k.shape}")
        object.__setattr__(self, "kernel", k)


def _normalise(kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float64)
    s = kernel.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("kernel must have positive finite mass")
    return kernel / s


def gaussian_kernel(sigma: float, size: int | None = None) -> np.ndarray:
    """Isotropic Gaussian PSF, normalised to unit mass."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if size is None:
        size = 2 * int(np.ceil(3.0 * sigma)) + 1
    if size % 2 == 0:
        size += 1
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    return _normalise(np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2)))


def disk_kernel(radius: float, size: int | None = None, supersample: int = 8) -> np.ndarray:
    """Anti-aliased defocus disk, normalised to unit mass."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    if size is None:
        size = 2 * int(np.ceil(radius)) + 1
    if size % 2 == 0:
        size += 1
    n = size * supersample
    coords = (np.arange(n, dtype=np.float64) + 0.5) / supersample - size / 2.0
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    mask = (xx**2 + yy**2) <= radius**2
    blocks = mask.reshape(size, supersample, size, supersample).mean(axis=(1, 3))
    return _normalise(blocks)


def motion_kernel(
    length: float,
    angle_deg: float,
    thickness: float = 1.5,
    size: int | None = None,
    supersample: int = 8,
) -> np.ndarray:
    """Anti-aliased line-segment motion PSF, normalised to unit mass."""
    if length <= 0 or thickness <= 0:
        raise ValueError("length and thickness must be positive")
    if size is None:
        size = 2 * int(np.ceil(length / 2.0 + thickness)) + 1
    if size % 2 == 0:
        size += 1
    n = size * supersample
    coords = (np.arange(n, dtype=np.float64) + 0.5) / supersample - size / 2.0
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    theta = np.deg2rad(angle_deg)
    # rotate into the line frame: u along the motion, v across it
    u = xx * np.cos(theta) + yy * np.sin(theta)
    v = -xx * np.sin(theta) + yy * np.cos(theta)
    mask = (np.abs(u) <= length / 2.0) & (np.abs(v) <= thickness / 2.0)
    blocks = mask.reshape(size, supersample, size, supersample).mean(axis=(1, 3))
    if blocks.sum() == 0:  # pragma: no cover - pathological parameters
        raise ValueError("motion kernel is empty; increase thickness or size")
    return _normalise(blocks)


def anisotropic_gaussian_kernel(
    sigma_x: float,
    sigma_y: float,
    angle_deg: float,
    size: int | None = None,
) -> np.ndarray:
    """Rotated anisotropic Gaussian PSF (``sigma_x`` along ``angle_deg``)."""
    if sigma_x <= 0 or sigma_y <= 0:
        raise ValueError("sigma_x and sigma_y must be positive")
    if size is None:
        size = 2 * int(np.ceil(3.0 * max(sigma_x, sigma_y))) + 1
    if size % 2 == 0:
        size += 1
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    theta = np.deg2rad(angle_deg)
    u = xx * np.cos(theta) + yy * np.sin(theta)
    v = -xx * np.sin(theta) + yy * np.cos(theta)
    return _normalise(np.exp(-0.5 * ((u / sigma_x) ** 2 + (v / sigma_y) ** 2)))


SHAKE_MAX_ANXIETY = 0.03


def trajectory_kernel(
    extent: float,
    intensity: float,
    seed: int,
    size: int | None = None,
    steps: int = 2000,
) -> np.ndarray:
    """Random-trajectory camera-shake PSF (Boracchi & Foi, 2012).

    A sensor trajectory is simulated as a random walk with inertia, a
    centripetal pull, Gaussian jitter and occasional abrupt shakes. Its
    ``intensity`` in (0, 1] scales the irregularity ("anxiety") linearly up to
    ``SHAKE_MAX_ANXIETY``: low values give nearly straight strokes, 1 gives
    curved, kinked paths like recorded hand shake (calibrated visually against
    the Levin kernels; larger anxiety produces unrealistic scribbles). The path is scaled so its largest extent is ``extent`` pixels and
    rasterised with bilinear splatting over the exposure. ``seed`` fixes the
    trajectory, so a kernel is reproducible from its parameters.
    """
    if extent <= 0:
        raise ValueError("extent must be positive")
    if not 0.0 < intensity <= 1.0:
        raise ValueError("intensity must be in (0, 1]")
    anxiety = SHAKE_MAX_ANXIETY * intensity
    rng = np.random.default_rng(seed)
    centripetal = 0.7 * rng.random()
    prob_big_shake = 0.2 * rng.random()
    gaussian_shake = 10.0 * rng.random()
    direction = np.exp(1j * 2.0 * np.pi * rng.random())
    unit = 1.0 / (steps - 1)
    velocity = direction * anxiety
    path = np.zeros(steps, dtype=np.complex128)
    for t in range(steps - 1):
        shake = 0.0
        if rng.random() < prob_big_shake * anxiety:
            shake = 2.0 * velocity * np.exp(1j * (np.pi + rng.random() - 0.5))
        jitter = gaussian_shake * complex(rng.standard_normal(), rng.standard_normal())
        velocity = velocity + shake + anxiety * (jitter - centripetal * path[t]) * unit
        velocity = velocity / abs(velocity) * unit
        path[t + 1] = path[t] + velocity

    span = max(np.ptp(path.real), np.ptp(path.imag))
    path = (path - complex(path.real.mean(), path.imag.mean())) * (extent / max(span, 1e-12))
    if size is None:
        size = 2 * int(np.ceil(extent / 2.0)) + 3
    if size % 2 == 0:
        size += 1
    centre = (size - 1) / 2.0
    xs, ys = path.real + centre, path.imag + centre
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = xs - x0, ys - y0
    kernel = np.zeros((size, size))
    for dx, dy, w in (
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ):
        np.add.at(kernel, (np.clip(y0 + dy, 0, size - 1), np.clip(x0 + dx, 0, size - 1)), w)
    return _normalise(kernel)


def airy_kernel(first_zero: float, size: int | None = None, supersample: int = 8) -> np.ndarray:
    """Diffraction-limited Airy pattern whose first dark ring has radius ``first_zero``."""
    from scipy.special import j1

    if first_zero <= 0:
        raise ValueError("first_zero must be positive")
    if size is None:
        size = 2 * int(np.ceil(3.0 * first_zero)) + 1
    if size % 2 == 0:
        size += 1
    n = size * supersample
    coords = (np.arange(n, dtype=np.float64) + 0.5) / supersample - size / 2.0
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    r = np.hypot(xx, yy) * (3.8317059702 / first_zero)  # first zero of J1
    with np.errstate(invalid="ignore", divide="ignore"):
        intensity = np.where(r > 1e-12, (2.0 * j1(r) / r) ** 2, 1.0)
    blocks = intensity.reshape(size, supersample, size, supersample).mean(axis=(1, 3))
    return _normalise(blocks)


def moffat_kernel(alpha: float, beta: float, size: int | None = None) -> np.ndarray:
    """Moffat profile ``(1 + (r/alpha)^2)^-beta``: heavy tails, as in atmospheric blur."""
    if alpha <= 0 or beta <= 1:
        raise ValueError("alpha must be positive and beta > 1")
    if size is None:
        # radius where the profile falls below 1% of its peak
        size = 2 * int(np.ceil(alpha * np.sqrt(100.0 ** (1.0 / beta) - 1.0))) + 1
    if size % 2 == 0:
        size += 1
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    return _normalise((1.0 + (xx**2 + yy**2) / alpha**2) ** (-beta))


def ring_kernel(outer: float, inner: float, size: int | None = None, supersample: int = 8) -> np.ndarray:
    """Anti-aliased annulus (mirror-lens / catadioptric defocus), unit mass."""
    if not 0 < inner < outer:
        raise ValueError("need 0 < inner < outer")
    if size is None:
        size = 2 * int(np.ceil(outer)) + 1
    if size % 2 == 0:
        size += 1
    n = size * supersample
    coords = (np.arange(n, dtype=np.float64) + 0.5) / supersample - size / 2.0
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    r2 = xx**2 + yy**2
    mask = (r2 <= outer**2) & (r2 >= inner**2)
    blocks = mask.reshape(size, supersample, size, supersample).mean(axis=(1, 3))
    return _normalise(blocks)


def ghost_kernel(separation: float, angle_deg: float, weight: float = 0.65) -> np.ndarray:
    """Double image (ghosting / double exposure): two points ``separation`` px apart.

    The main point carries ``weight`` of the mass, the ghost the rest; both are
    placed symmetrically about the centre with bilinear splatting.
    """
    if separation <= 0 or not 0.5 <= weight < 1.0:
        raise ValueError("separation must be positive and weight in [0.5, 1)")
    size = 2 * int(np.ceil(separation / 2.0)) + 3
    centre = (size - 1) / 2.0
    theta = np.deg2rad(angle_deg)
    kernel = np.zeros((size, size))
    for sign, w in ((-1.0, weight), (1.0, 1.0 - weight)):
        x = centre + sign * separation / 2.0 * np.cos(theta)
        y = centre + sign * separation / 2.0 * np.sin(theta)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        kernel[y0, x0] += w * (1 - fx) * (1 - fy)
        kernel[y0, x0 + 1] += w * fx * (1 - fy)
        kernel[y0 + 1, x0] += w * (1 - fx) * fy
        kernel[y0 + 1, x0 + 1] += w * fx * fy
    return _normalise(kernel)


def kernel_axes(kernel: np.ndarray) -> tuple[float, float]:
    """Standard deviations (minor, major) of the kernel mass along its principal axes."""
    k = kernel / kernel.sum()
    y, x = np.indices(k.shape, dtype=np.float64)
    mx, my = (k * x).sum(), (k * y).sum()
    dx, dy = x - mx, y - my
    cov = np.array([[(k * dx * dx).sum(), (k * dx * dy).sum()], [(k * dx * dy).sum(), (k * dy * dy).sum()]])
    minor, major = np.sqrt(np.clip(np.linalg.eigvalsh(cov), 0.0, None))
    return float(minor), float(major)


def fit_support(kernel: np.ndarray, max_size: int, threshold: float = 1e-3) -> np.ndarray:
    """Crop ``kernel`` to its centred support and cap it at ``max_size`` x ``max_size``.

    The crop is symmetric about the kernel centre so the PSF origin is kept.
    Mass outside the cap is dropped and the result renormalised.
    """
    kernel = _normalise(kernel)
    rows, cols = np.nonzero(kernel >= threshold * kernel.max())
    h, w = kernel.shape
    cy, cx = h // 2, w // 2
    half_h = min(max(abs(rows - cy).max(), 1), (max_size - 1) // 2)
    half_w = min(max(abs(cols - cx).max(), 1), (max_size - 1) // 2)
    half = max(half_h, half_w)
    top, left = max(cy - half, 0), max(cx - half, 0)
    return _normalise(kernel[top : cy + half + 1, left : cx + half + 1])


def default_motion_kernels(
    lengths: tuple[float, ...] = (9.0, 15.0),
    angles: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0),
    thickness: float = 1.5,
) -> list[PSF]:
    """Procedural motion kernels used when the Levin files are unavailable."""
    return [
        PSF(
            name=f"motion_l{int(length)}_a{int(angle)}",
            kernel=motion_kernel(length, angle, thickness=thickness),
            kind="motion",
        )
        for length in lengths
        for angle in angles
    ]


def _load_kernel_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    if suffix == ".mat":
        from scipy.io import loadmat

        data = loadmat(path)
        candidates = [
            v
            for v in data.values()
            if isinstance(v, np.ndarray)
            and v.ndim == 2
            and np.issubdtype(v.dtype, np.number)
        ]
        if not candidates:
            raise ValueError(f"no 2D numeric array found in {path}")
        return np.asarray(candidates[0], dtype=np.float64)
    if Image is None:  # pragma: no cover
        raise ImportError("Pillow is required to read image kernels")
    with Image.open(path) as im:
        return np.asarray(im.convert("F"), dtype=np.float64)


def load_kernel_dir(
    directory: str | Path,
    kind: str = "real",
    prefix: str = "",
    max_size: int | None = None,
) -> list[PSF]:
    """Load every kernel file in ``directory`` (sorted by name).

    Accepts ``.npy``, ``.mat`` and common image formats. With ``max_size``,
    each kernel is cropped to its support and capped (see :func:`fit_support`).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"kernel directory not found: {directory}")
    suffixes = {".npy", ".mat", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in suffixes)
    if not paths:
        raise FileNotFoundError(f"no kernel files found in {directory}")
    psfs = []
    for path in paths:
        kernel = _normalise(_load_kernel_file(path))
        if max_size is not None:
            kernel = fit_support(kernel, max_size)
        psfs.append(PSF(name=f"{prefix}{path.stem}", kernel=kernel, kind=kind))
    return psfs


def load_levin_kernels(directory: str | Path) -> list[PSF]:
    """Load the official Levin et al. motion kernels from ``directory``.

    Accepts ``.npy``, ``.mat`` and common image formats. Files are sorted by
    name so the kernel order is deterministic.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Levin kernel directory not found: {directory}")
    suffixes = {".npy", ".mat", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in suffixes)
    if not paths:
        raise FileNotFoundError(f"no kernel files found in {directory}")
    return [
        PSF(name=p.stem, kernel=_normalise(_load_kernel_file(p)), kind="motion")
        for p in paths
    ]


PROTOCOL_VERSIONS = ("v1", "v2")
V2_MAX_KERNEL_SIZE = 31


def protocol_psf_sets(
    version: str = "v1",
    *,
    real_kernel_dirs: tuple[tuple[str, str], ...] = (),
    train_gaussian_sigmas: tuple[float, ...] = (0.7, 1.2, 1.6, 2.0),
    interp_gaussian_sigmas: tuple[float, ...] = (0.9, 1.4, 1.8),
    train_motion_lengths: tuple[float, ...] = (9.0, 15.0),
    train_motion_angles: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0),
    interp_motion_lengths: tuple[float, ...] = (11.0, 13.0),
    interp_motion_angles: tuple[float, ...] = (22.5, 67.5, 112.5, 157.5),
    extrap_defocus_radii: tuple[float, ...] = (3.0, 5.0, 7.0, 9.0),
    thickness: float = 1.5,
) -> dict[str, list[PSF]]:
    """Locked P1 / P2 kernel sets for the controlled protocol.

    P1 (training, 12 kernels): 4 Gaussian + 8 motion.
    P2-interp (11): unseen parameters of the same families, strictly inside
    the P1 parameter ranges so it tests interpolation, not extrapolation.
    P2-extrap (4): defocus disks, a family absent from training.

    The three sets are disjoint by construction (different names and
    parameters). P2-interp tests interpolation within seen families;
    P2-extrap tests extrapolation to an unseen family.

    ``version="v2"`` returns the expanded protocol instead (see
    :func:`protocol_v2_psf_sets`); the keyword parameters apply to v1 only and
    ``real_kernel_dirs`` to v2 only. v1 is frozen: its sets never change.
    """
    if version == "v2":
        return protocol_v2_psf_sets(real_kernel_dirs)
    if version != "v1":
        raise ValueError(f"unknown protocol version {version!r}; choose from {PROTOCOL_VERSIONS}")
    p1: list[PSF] = [
        PSF(name=f"gaussian_sigma{sigma:g}", kernel=gaussian_kernel(sigma), kind="gaussian")
        for sigma in train_gaussian_sigmas
    ]
    p1 += [
        PSF(
            name=f"motion_l{length:g}_a{angle:g}",
            kernel=motion_kernel(length, angle, thickness=thickness),
            kind="motion",
        )
        for length in train_motion_lengths
        for angle in train_motion_angles
    ]

    p2_interp: list[PSF] = [
        PSF(name=f"gaussian_sigma{sigma:g}", kernel=gaussian_kernel(sigma), kind="gaussian")
        for sigma in interp_gaussian_sigmas
    ]
    p2_interp += [
        PSF(
            name=f"motion_l{length:g}_a{angle:g}",
            kernel=motion_kernel(length, angle, thickness=thickness),
            kind="motion",
        )
        for length in interp_motion_lengths
        for angle in interp_motion_angles
    ]

    p2_extrap: list[PSF] = [
        PSF(name=f"defocus_r{radius:g}", kernel=disk_kernel(radius), kind="defocus")
        for radius in extrap_defocus_radii
    ]

    sets = {"P1": p1, "P2_interp": p2_interp, "P2_extrap": p2_extrap}
    names = [psf.name for kernels in sets.values() for psf in kernels]
    if len(names) != len(set(names)):
        raise ValueError("PSF protocol sets contain duplicate names")
    return sets


def _pick(items: list[PSF], count: int, seed: int) -> list[PSF]:
    """``count`` items drawn without replacement with a fixed seed, in pool order."""
    index = np.sort(np.random.default_rng(seed).choice(len(items), size=count, replace=False))
    return [items[i] for i in index]


def _v2_psf(name: str, kernel: np.ndarray, kind: str) -> PSF:
    return PSF(name=name, kernel=fit_support(kernel, V2_MAX_KERNEL_SIZE), kind=kind)


# Family separation margins for v2 (no family is a special or limiting case of another):
V2_ELLIPSES = ((3.6, 1.2), (4.2, 1.2), (4.8, 1.2), (4.5, 1.5))  # (sigma_x, sigma_y): aspect >= 3, sigma_y >= 1.2
V2_MAX_CROSS_FAMILY_CORR = 0.90  # best-shift normalised correlation between kernels of different families
V2_MIN_SHAKE_MINOR_STD = 1.2  # shake paths: cross-axis spread >= 1.2 px (straight lines: <= 0.61 px)
V2_RING_INNER_RATIO = 0.6  # rings: inner radius = 0.6 x outer, a clear hole (never a disc)


def _shake_family(intensities, count: int, seed_start: int) -> list[PSF]:
    """``count`` curved shake kernels per intensity; near-straight paths are rejected."""
    out, seed = [], seed_start
    for i in intensities:
        accepted = 0
        while accepted < count:
            kernel = trajectory_kernel(_shake_extent(seed), i, seed)
            if kernel_axes(kernel)[0] >= V2_MIN_SHAKE_MINOR_STD:
                out.append(_v2_psf(f"shake_i{i:g}_s{seed}", kernel, "shake"))
                accepted += 1
            seed += 1
    return out


def _v2_family_pools() -> dict[str, list[PSF]]:
    """Protocol v2 training pool, by family (training grids only)."""
    gaussian = [
        _v2_psf(f"gaussian_sigma{s:g}", gaussian_kernel(s), "gaussian")
        for s in np.round(np.arange(0.6, 2.41, 0.2), 1)
    ]
    aniso = [
        _v2_psf(f"aniso_sx{sx:g}_sy{sy:g}_a{a:g}", anisotropic_gaussian_kernel(sx, sy, a), "aniso_gaussian")
        for sx, sy in V2_ELLIPSES
        for a in range(0, 180, 15)
    ]
    motion = [
        _v2_psf(f"motion_l{l:g}_a{a:g}", motion_kernel(l, a, thickness=1.5), "motion")
        for l in (5, 9, 13, 17, 21)
        for a in range(0, 180, 15)
    ]
    shake = _shake_family((0.2, 0.4, 0.6, 0.8, 1.0), count=20, seed_start=0)
    return {"gaussian": gaussian, "aniso_gaussian": aniso, "motion": motion, "shake": shake}


def _shake_extent(seed: int) -> float:
    """Trajectory extent in pixels, 9-25, fixed by the seed."""
    return 9.0 + 16.0 * np.random.default_rng(10_000 + seed).random()


def protocol_v2_psf_sets(real_kernel_dirs: tuple[tuple[str, str], ...] = ()) -> dict[str, list[PSF]]:
    """Protocol v2 kernel sets (expanded families; see the spec, section 12.3).

    - ``P1_pool`` (training, about 290 kernels): isotropic Gaussian (0.2 grid in
      [0.6, 2.4]), elliptical Gaussian (``V2_ELLIPSES``, 15-degree angles),
      linear motion (lengths {5,9,13,17,21},
      15-degree angles) and curved random-trajectory shake (intensities
      {0.2,...,1.0}, 20 accepted seeds each; near-straight paths rejected).
    - ``P1`` (in-distribution test, 24): 6 kernels per family from the pool.
    - ``P2_interp`` (24): the same families between the training grid points
      (7.5-degree angle offsets, intermediate widths/lengths/intensities,
      held-out trajectory seeds).
    - ``P2_extrap``: unseen families: defocus disks, rings, ghosts and any real
      recorded kernels found in ``real_kernel_dirs`` (``(prefix, directory)``
      pairs, e.g. ``(("levin", "data/levin"),)``).

    Every kernel is cropped to its support and capped at 31 x 31.

    No family is a special or limiting case of another, and none is a near
    copy: elliptical Gaussians have aspect >= 3 and sigma_y >= 1.2 (thinner
    ones are motion lines, rounder ones round Gaussians), shake paths must be
    curved (``V2_MIN_SHAKE_MINOR_STD``), rings have a hole, ghosts are two
    separated points, and the smallest defocus disc (radius 3) is left out.
    Moffat (beta -> infinity gives a Gaussian) and Airy (its faint rings leave
    a Gaussian, correlation 0.99) are not used. The construction checks that
    every cross-family pair stays at or below ``V2_MAX_CROSS_FAMILY_CORR``.
    """
    pools = _v2_family_pools()
    p1_pool = [psf for family in pools.values() for psf in family]
    p1 = [psf for i, family in enumerate(pools.values()) for psf in _pick(family, 6, seed=i)]

    interp = [
        _v2_psf(f"gaussian_sigma{s:g}", gaussian_kernel(s), "gaussian")
        for s in (0.7, 1.1, 1.3, 1.7, 2.1, 2.3)
    ]
    interp += [
        _v2_psf(f"aniso_sx{sx:g}_sy{sy:g}_a{a:g}", anisotropic_gaussian_kernel(sx, sy, a), "aniso_gaussian")
        for sx, sy, a in ((3.9, 1.2, 7.5), (4.5, 1.2, 52.5), (4.2, 1.35, 97.5), (3.9, 1.2, 142.5),
                          (4.5, 1.2, 22.5), (4.2, 1.35, 112.5))
    ]
    interp += [
        _v2_psf(f"motion_l{l:g}_a{a:g}", motion_kernel(l, a, thickness=1.5), "motion")
        for l, a in ((7, 7.5), (11, 37.5), (15, 67.5), (19, 97.5), (11, 127.5), (15, 157.5))
    ]
    interp += _shake_family((0.3, 0.5, 0.7, 0.9, 0.5, 0.7), count=1, seed_start=1_000_000)  # far from pool seeds

    extrap = [_v2_psf(f"defocus_r{r:g}", disk_kernel(r), "defocus") for r in (5, 7, 9, 11)]
    extrap += [
        _v2_psf(f"ring_r{r:g}", ring_kernel(r, V2_RING_INNER_RATIO * r), "ring") for r in (4, 6, 8, 10)
    ]
    extrap += [
        _v2_psf(f"ghost_d{d:g}_a{a:g}", ghost_kernel(d, a), "ghost")
        for d, a in ((6, 30), (10, 100), (14, 160), (18, 60))
    ]
    for prefix, directory in real_kernel_dirs:
        extrap += load_kernel_dir(directory, kind="real", prefix=f"{prefix}_", max_size=V2_MAX_KERNEL_SIZE)

    sets = {"P1_pool": p1_pool, "P1": p1, "P2_interp": interp, "P2_extrap": extrap}
    pool_names = {psf.name for psf in p1_pool}
    if len(pool_names) != len(p1_pool):
        raise ValueError("v2 training pool contains duplicate names")
    if not {psf.name for psf in p1} <= pool_names:
        raise ValueError("v2 P1 test kernels must come from the training pool")
    pool_seeds = {psf.name.rsplit("_s", 1)[1] for psf in p1_pool if psf.kind == "shake"}
    if any(psf.name.rsplit("_s", 1)[1] in pool_seeds for psf in interp if psf.kind == "shake"):
        raise ValueError("v2 held-out shake kernels reuse a training trajectory seed")
    held_out = [psf.name for psf in interp + extrap]
    if len(held_out) != len(set(held_out)) or set(held_out) & pool_names:
        raise ValueError("v2 held-out kernels overlap the training pool or each other")
    corr, a, b = max_cross_family_correlation([psf for psf in p1_pool + interp + extrap if psf.kind != "real"])
    if corr > V2_MAX_CROSS_FAMILY_CORR:
        raise ValueError(f"v2 families overlap: {a} ~ {b} (correlation {corr:.3f})")
    return sets


def max_cross_family_correlation(psfs: list[PSF], size: int = 64) -> tuple[float, str, str]:
    """Largest best-shift normalised correlation between two kernels of different kinds.

    1 means the same shape up to a shift; v2 keeps every cross-family pair at or
    below ``V2_MAX_CROSS_FAMILY_CORR`` so no family is a near copy of another.
    """
    spectra = []
    for psf in psfs:
        padded = np.zeros((size, size))
        h, w = psf.kernel.shape
        padded[:h, :w] = psf.kernel / np.linalg.norm(psf.kernel)
        spectra.append(np.fft.rfft2(padded))
    spectra = np.stack(spectra)
    kinds = np.array([psf.kind for psf in psfs])
    best = (0.0, "", "")
    for i in range(len(psfs)):
        other = np.flatnonzero(kinds[i + 1 :] != kinds[i]) + i + 1
        if other.size == 0:
            continue
        corr = np.fft.irfft2(spectra[other] * np.conj(spectra[i])[None], s=(size, size)).max(axis=(1, 2))
        j = int(corr.argmax())
        if corr[j] > best[0]:
            best = (float(corr[j]), psfs[i].name, psfs[other[j]].name)
    return best


def build_psf_library(
    gaussian_sigmas: tuple[float, ...] = (0.7, 1.2, 1.6, 2.0),
    defocus_radii: tuple[float, ...] = (3.0, 5.0, 7.0, 9.0),
    motion_kernels: list[PSF] | None = None,
    levin_dir: str | Path | None = None,
) -> list[PSF]:
    """Build the shared PSF library for the protocol.

    If ``motion_kernels`` is omitted, the official Levin set is loaded when
    ``levin_dir`` is given; otherwise procedural motion kernels are used.
    """
    psfs: list[PSF] = []
    for sigma in gaussian_sigmas:
        psfs.append(
            PSF(name=f"gaussian_sigma{sigma:g}", kernel=gaussian_kernel(sigma), kind="gaussian")
        )
    for radius in defocus_radii:
        psfs.append(
            PSF(name=f"defocus_r{radius:g}", kernel=disk_kernel(radius), kind="defocus")
        )
    if motion_kernels is None:
        motion_kernels = (
            load_levin_kernels(levin_dir) if levin_dir is not None else default_motion_kernels()
        )
    psfs.extend(motion_kernels)
    return psfs
