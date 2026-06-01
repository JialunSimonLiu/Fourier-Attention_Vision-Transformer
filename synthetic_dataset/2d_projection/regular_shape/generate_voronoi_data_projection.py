################################################################################
# generate_voronoi_data_projection.py
# Generate one projected 10-domain Voronoi crystal and its diffraction pattern.
# - Jialun Liu, LCN, UCL, 05.2026, jialun.liu.17@ucl.ac.uk
#
# Input:
#   None. The crystal is generated from fixed simulation parameters and seed.
#
# Output:
#   regular_10_domain/amp.tif
#   regular_10_domain/phase.tif
#   regular_10_domain/dp_amp.tif
#   regular_10_domain/support_original.tif
#   regular_10_domain/support_square.tif
################################################################################

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import tifffile


# ============================================================
# User parameters
# ============================================================

OUT_DIR = Path("regular_10_domain")

N = 64
NUM_DOMAINS = 10

SEED0 = 1234
CRYSTAL_SEED = SEED0 + 4

MIN_ADJACENT_PHASE_DIFF = 0.5
PHASE_LOW = -np.pi
PHASE_HIGH = np.pi

SPHERE_RADIUS = 16.0

# Use "cylinder_z" if you want the projected XY object to show a real side bite.
# Use "sphere" if you want a true spherical bite, but then the XY projection
# may look partly filled.
CUT_GEOMETRY = "cylinder_z"   # "cylinder_z" or "sphere"

CUT_RADIUS = 5.0
CUT_OFFSET_X = 14.0
CUT_OFFSET_Y = 0.0
CUT_OFFSET_Z = 0.0

SOFT_SIGMA = 1.35


# ============================================================
# Basic utilities
# ============================================================

def normalise_max(arr, eps=1e-12):
    arr = np.asarray(arr, dtype=np.float32)
    m = np.nanmax(arr)
    if not np.isfinite(m) or m < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / m).astype(np.float32)


def wrap_phase_difference(a, b):
    return np.angle(np.exp(1j * (a - b)))


def circular_distance(a, b):
    return abs(wrap_phase_difference(a, b))


def save_float32_tif(path, arr):
    tifffile.imwrite(str(path), np.asarray(arr, dtype=np.float32))


# ============================================================
# 3D object generation
# ============================================================

def make_coordinate_grid(n):
    """
    Internal array convention:
        array[z, y, x]
    """
    coords = np.arange(n, dtype=np.float32) - (n - 1) / 2.0
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")
    return z, y, x


def make_3d_support(
    n=N,
    sphere_radius=SPHERE_RADIUS,
    cut_radius=CUT_RADIUS,
    cut_offset_x=CUT_OFFSET_X,
    cut_offset_y=CUT_OFFSET_Y,
    cut_offset_z=CUT_OFFSET_Z,
    cut_geometry=CUT_GEOMETRY,
):
    """
    3D support:
        main sphere minus side cut.

    cut_geometry = "sphere":
        A true spherical bite. This is physically 3D, but the XY projection
        can be partly filled by material at other z positions.

    cut_geometry = "cylinder_z":
        A cylindrical cut running through z. This gives the intended XY
        projected side bite shape.
    """
    z, y, x = make_coordinate_grid(n)

    main_sphere = (x**2 + y**2 + z**2) <= sphere_radius**2

    if cut_geometry == "sphere":
        cut_region = (
            (x - cut_offset_x) ** 2
            + (y - cut_offset_y) ** 2
            + (z - cut_offset_z) ** 2
        ) <= cut_radius**2

    elif cut_geometry == "cylinder_z":
        cut_region = (
            (x - cut_offset_x) ** 2
            + (y - cut_offset_y) ** 2
        ) <= cut_radius**2

    else:
        raise ValueError(
            "CUT_GEOMETRY must be either 'sphere' or 'cylinder_z'."
        )

    support = main_sphere & (~cut_region)
    return support.astype(bool)


def make_soft_amplitude_from_support(support, sigma=SOFT_SIGMA):
    """
    Soft-edged 3D amplitude, support-masked.
    """
    support_float = support.astype(np.float32)

    amp = gaussian_filter(support_float, sigma=sigma)
    amp = normalise_max(amp)

    amp = amp * support_float
    amp = normalise_max(amp)

    return amp.astype(np.float32)


def sample_seed_points_inside_support(support, num_domains, rng):
    points = np.argwhere(support)
    if len(points) < num_domains:
        raise ValueError("Support contains fewer voxels than NUM_DOMAINS.")

    selected = rng.choice(len(points), size=num_domains, replace=False)
    seeds = points[selected]
    return seeds.astype(np.float32)


def assign_3d_voronoi_labels(support, seeds):
    labels = -np.ones(support.shape, dtype=np.int16)

    support_points = np.argwhere(support).astype(np.float32)

    tree = cKDTree(seeds)
    _, nearest = tree.query(support_points, k=1)

    labels[tuple(support_points.astype(int).T)] = nearest.astype(np.int16)

    return labels


def compute_domain_adjacency(labels, num_domains):
    adjacency = set()

    for axis in range(3):
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3

        sl_a[axis] = slice(0, labels.shape[axis] - 1)
        sl_b[axis] = slice(1, labels.shape[axis])

        a = labels[tuple(sl_a)]
        b = labels[tuple(sl_b)]

        mask = (a >= 0) & (b >= 0) & (a != b)
        pairs = np.stack([a[mask], b[mask]], axis=1)

        for i, j in pairs:
            i = int(i)
            j = int(j)
            if i > j:
                i, j = j, i
            adjacency.add((i, j))

    neighbours = {i: set() for i in range(num_domains)}

    for i, j in adjacency:
        neighbours[i].add(j)
        neighbours[j].add(i)

    return neighbours


def assign_domain_phases_with_threshold(
    neighbours,
    num_domains,
    rng,
    min_diff=MIN_ADJACENT_PHASE_DIFF,
    phase_low=PHASE_LOW,
    phase_high=PHASE_HIGH,
    max_trials_per_domain=10000,
):
    phases = np.full(num_domains, np.nan, dtype=np.float32)

    order = sorted(
        range(num_domains),
        key=lambda i: len(neighbours[i]),
        reverse=True,
    )

    for domain in order:
        assigned_neighbours = [
            j for j in neighbours[domain] if np.isfinite(phases[j])
        ]

        accepted = False

        for _ in range(max_trials_per_domain):
            candidate = rng.uniform(phase_low, phase_high)

            ok = True
            for j in assigned_neighbours:
                if circular_distance(candidate, phases[j]) < min_diff:
                    ok = False
                    break

            if ok:
                phases[domain] = candidate
                accepted = True
                break

        if not accepted:
            raise RuntimeError(
                f"Could not assign phase for domain {domain}. "
                f"Try reducing MIN_ADJACENT_PHASE_DIFF."
            )

    return phases.astype(np.float32)


def make_3d_phase(labels, domain_phases):
    phase = np.zeros(labels.shape, dtype=np.float32)

    inside = labels >= 0
    phase[inside] = domain_phases[labels[inside]]

    return phase


def verify_adjacency_threshold(neighbours, domain_phases, min_diff):
    values = []

    for i, js in neighbours.items():
        for j in js:
            if i < j:
                values.append(circular_distance(domain_phases[i], domain_phases[j]))

    if len(values) == 0:
        return np.nan

    min_value = float(np.min(values))

    if min_value + 1e-7 < min_diff:
        raise RuntimeError(
            f"Adjacent phase threshold failed: minimum = {min_value:.4f} rad, "
            f"required = {min_diff:.4f} rad"
        )

    return min_value


def generate_one_3d_crystal(seed):
    rng = np.random.default_rng(seed)

    support = make_3d_support()
    amp = make_soft_amplitude_from_support(support)

    seeds = sample_seed_points_inside_support(support, NUM_DOMAINS, rng)
    labels = assign_3d_voronoi_labels(support, seeds)

    neighbours = compute_domain_adjacency(labels, NUM_DOMAINS)

    domain_phases = assign_domain_phases_with_threshold(
        neighbours=neighbours,
        num_domains=NUM_DOMAINS,
        rng=rng,
        min_diff=MIN_ADJACENT_PHASE_DIFF,
    )

    min_adjacent_diff = verify_adjacency_threshold(
        neighbours=neighbours,
        domain_phases=domain_phases,
        min_diff=MIN_ADJACENT_PHASE_DIFF,
    )

    phase = make_3d_phase(labels, domain_phases)

    rho = amp.astype(np.float32) * np.exp(1j * phase.astype(np.float32))
    rho = rho.astype(np.complex64)

    return {
        "support": support,
        "amplitude": amp,
        "phase": phase,
        "rho": rho,
        "labels": labels,
        "seeds": seeds,
        "domain_phases": domain_phases,
        "min_adjacent_diff": min_adjacent_diff,
    }


# ============================================================
# Corrected 2D XY projection and diffraction
# ============================================================

def project_3d_object_to_xy(crystal, eps=1e-8):
    """
    Project the full 3D complex object along Z onto the XY plane.

    Correct coherent projection:
        rho_proj[y, x] = sum_z rho_3d[z, y, x]

    Do NOT average amplitude and phase separately.
    The projected amplitude and phase must be extracted only after
    the complex projection:
        amp_proj   = |rho_proj|
        phase_proj = angle(rho_proj)
    """
    rho_3d = np.asarray(crystal["rho"], dtype=np.complex64)
    support_3d = np.asarray(crystal["support"], dtype=bool)

    if rho_3d.ndim != 3:
        raise ValueError(f"rho_3d must be 3D [z, y, x], got shape {rho_3d.shape}")

    if support_3d.shape != rho_3d.shape:
        raise ValueError(
            f"support_3d shape {support_3d.shape} does not match rho_3d shape {rho_3d.shape}"
        )

    # Coherent projection along z.
    # Input:  [z, y, x]
    # Output: [y, x]
    rho_proj_raw = np.sum(rho_3d, axis=0).astype(np.complex64)

    # Geometric projected support.
    support_proj = np.any(support_3d, axis=0)

    if rho_proj_raw.shape != support_proj.shape:
        raise ValueError(
            f"Projection shape mismatch: rho_proj {rho_proj_raw.shape}, "
            f"support_proj {support_proj.shape}"
        )

    # Projected amplitude before normalisation.
    amp_raw = np.abs(rho_proj_raw).astype(np.float32)
    amp_raw = amp_raw * support_proj.astype(np.float32)

    # Normalise amplitude for saving/training.
    amp_proj = normalise_max(amp_raw)
    amp_proj = amp_proj * support_proj.astype(np.float32)

    # Phase should only be trusted where projected amplitude is non-zero.
    phase_proj = np.zeros_like(amp_proj, dtype=np.float32)
    valid_phase = support_proj & (amp_raw > eps)
    phase_proj[valid_phase] = np.angle(rho_proj_raw[valid_phase]).astype(np.float32)

    # Rebuild the saved projected object from saved amp and phase.
    # This changes only the global amplitude scale, not the phase geometry.
    rho_proj_saved = amp_proj * np.exp(1j * phase_proj)
    rho_proj_saved = rho_proj_saved * support_proj.astype(np.float32)
    rho_proj_saved = rho_proj_saved.astype(np.complex64)

    # Final sanity checks.
    assert amp_proj.shape == (N, N), f"amp_proj shape is {amp_proj.shape}, expected {(N, N)}"
    assert phase_proj.shape == (N, N), f"phase_proj shape is {phase_proj.shape}, expected {(N, N)}"
    assert support_proj.shape == (N, N), f"support_proj shape is {support_proj.shape}, expected {(N, N)}"
    assert rho_proj_saved.shape == (N, N), f"rho_proj shape is {rho_proj_saved.shape}, expected {(N, N)}"

    return {
        "rho": rho_proj_saved,
        "amp": amp_proj.astype(np.float32),
        "phase": phase_proj.astype(np.float32),
        "support": support_proj.astype(bool),
        "rho_raw": rho_proj_raw.astype(np.complex64),
        "amp_raw": amp_raw.astype(np.float32),
    }


def compute_diffraction_amplitude(rho_2d):
    if rho_2d.shape != (N, N):
        raise ValueError(f"rho_2d has wrong shape: {rho_2d.shape}, expected {(N, N)}")

    dp_complex = np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(rho_2d),
            norm="ortho",
        )
    )

    dp_amp = np.abs(dp_complex).astype(np.float32)
    dp_amp = normalise_max(dp_amp)

    return dp_amp


def make_square_support_32(n=N, square_size=32):
    support = np.zeros((n, n), dtype=np.float32)

    start = (n - square_size) // 2
    end = start + square_size

    support[start:end, start:end] = 1.0

    return support


# ============================================================
# Main
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    crystal = generate_one_3d_crystal(seed=CRYSTAL_SEED)

    xy_projection = project_3d_object_to_xy(crystal)

    amp_2d = xy_projection["amp"]
    phase_2d = xy_projection["phase"]
    support_2d = xy_projection["support"]
    rho_2d = xy_projection["rho"]

    if amp_2d.shape != (N, N):
        raise RuntimeError(f"amp_2d wrong shape: {amp_2d.shape}")

    if phase_2d.shape != (N, N):
        raise RuntimeError(f"phase_2d wrong shape: {phase_2d.shape}")

    if support_2d.shape != (N, N):
        raise RuntimeError(f"support_2d wrong shape: {support_2d.shape}")

    if rho_2d.shape != (N, N):
        raise RuntimeError(f"rho_2d wrong shape: {rho_2d.shape}")

    dp_amp = compute_diffraction_amplitude(rho_2d)

    if dp_amp.shape != (N, N):
        raise RuntimeError(f"dp_amp wrong shape: {dp_amp.shape}")

    save_float32_tif(OUT_DIR / "amp.tif", amp_2d)
    save_float32_tif(OUT_DIR / "phase.tif", phase_2d)
    save_float32_tif(OUT_DIR / "dp_amp.tif", dp_amp)
    save_float32_tif(OUT_DIR / "support_original.tif", support_2d.astype(np.float32))
    save_float32_tif(OUT_DIR / "support_square.tif", make_square_support_32())


if __name__ == "__main__":
    main()