################################################################################
# generate_voronoi_data_projection.py
# Generate one irregular 10-domain projected crystal with soft phase boundaries.
# - Jialun Liu, LCN, UCL, 05.2026, jialun.liu.17@ucl.ac.uk
#
# Input:
#   None. The crystal is generated from fixed simulation parameters and seed.
#
# Output:
#   irregular_10_domain/amp.tif
#   irregular_10_domain/phase.tif
#   irregular_10_domain/dp_amp.tif
#   irregular_10_domain/support_original.tif
################################################################################

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import tifffile


# ============================================================
# User parameters
# ============================================================

OUT_DIR = Path("irregular_10_domain")

N = 64
NUM_DOMAINS = 10
SEED = 1234

# Crystal should span roughly within 32 pixels.
# Max effective radius is kept close to 16.
BASE_RADIUS = 12 #14
IRREGULAR_STRENGTH = 6.0 #6
IRREGULAR_SIGMA = 6.0 #7

# Soft amplitude setting, same as your old sphere/cylindrical-cut code.
SOFT_SIGMA = 1.35

# Domain phase settings.
MIN_ADJACENT_PHASE_DIFF = 0.5
PHASE_LOW = -np.pi
PHASE_HIGH = np.pi

# Soft phase wall setting.
# Increase to make smoother domain-wall transition.
PHASE_WALL_SIGMA = 2.0


# ============================================================
# Basic utilities
# ============================================================

def normalise_max(arr, eps=1e-12):
    arr = np.asarray(arr, dtype=np.float32)
    m = np.nanmax(arr)
    if not np.isfinite(m) or m < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / m).astype(np.float32)


def save_float32_tif(path, arr):
    tifffile.imwrite(str(path), np.asarray(arr, dtype=np.float32))


def wrap_phase_difference(a, b):
    return np.angle(np.exp(1j * (a - b)))


def circular_distance(a, b):
    return abs(wrap_phase_difference(a, b))


def make_coordinate_grid(n=N):
    """
    Internal array convention:
        array[z, y, x]
    """
    coords = np.arange(n, dtype=np.float32) - (n - 1) / 2.0
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")
    return z, y, x


# ============================================================
# Irregular 3D support and soft amplitude
# ============================================================

def make_irregular_levelset(seed):
    """
    Slightly irregular compact 3D crystal.

    The crystal is still approximately spherical and stays within
    roughly 32 pixels across, but the boundary is no longer perfectly round.

    levelset < 0: inside crystal
    levelset > 0: outside crystal
    """
    rng = np.random.default_rng(seed)

    z, y, x = make_coordinate_grid(N)
    r = np.sqrt(x**2 + y**2 + z**2)

    noise = rng.normal(size=(N, N, N)).astype(np.float32)
    noise = gaussian_filter(noise, sigma=IRREGULAR_SIGMA)
    noise = noise - np.mean(noise)
    noise = noise / (np.std(noise) + 1e-8)

    # Keep perturbation bounded so the crystal remains within ~32 pixels.
    perturb = IRREGULAR_STRENGTH * np.tanh(noise)

    effective_radius = BASE_RADIUS + perturb
    levelset = r - effective_radius

    return levelset.astype(np.float32)


def make_irregular_3d_support(seed):
    """
    Irregular 3D support.

    This replaces the old sphere/cylindrical-cut support with an irregular
    compact support, but keeps the same later amplitude-generation logic.
    """
    levelset = make_irregular_levelset(seed)
    support = levelset <= 0.0

    return support.astype(bool)


def make_soft_amplitude_from_support(support, sigma=SOFT_SIGMA):
    """
    Soft-edged 3D amplitude, support-masked.

    This is the same amplitude construction as your old sphere/cylindrical-cut
    code: smooth the binary support, then mask it back inside the hard support.
    """
    support_float = support.astype(np.float32)

    amp = gaussian_filter(support_float, sigma=sigma)
    amp = normalise_max(amp)

    amp = amp * support_float
    amp = normalise_max(amp)

    return amp.astype(np.float32)


def make_support_and_soft_amplitude(seed):
    """
    Hard irregular support plus old-style soft 3D amplitude.

    support:
        used for Voronoi seed placement, masking, projection support,
        and plotting.

    amplitude:
        generated in the same way as your old code:
        Gaussian-smoothed support, then support-masked.
    """
    support = make_irregular_3d_support(seed)
    amp = make_soft_amplitude_from_support(support, sigma=SOFT_SIGMA)

    # Keep this key for compatibility with the newer plotting/saving code.
    # It is now exactly the hard support, as in your old code.
    support_for_plot = support.copy()

    return support.astype(bool), amp.astype(np.float32), support_for_plot.astype(bool)


# ============================================================
# 3D Voronoi domains and phase
# ============================================================

def sample_seed_points_inside_support(support, num_domains, rng):
    points = np.argwhere(support)
    if len(points) < num_domains:
        raise ValueError("Support contains fewer voxels than NUM_DOMAINS.")

    selected = rng.choice(len(points), size=num_domains, replace=False)
    seeds = points[selected]

    return seeds.astype(np.float32)


def assign_3d_voronoi_labels(support, seeds):
    labels = -np.ones(support.shape, dtype=np.int16)

    points = np.argwhere(support).astype(np.float32)

    tree = cKDTree(seeds)
    _, nearest = tree.query(points, k=1)

    labels[tuple(points.astype(int).T)] = nearest.astype(np.int16)

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


def make_sharp_phase(labels, domain_phases):
    phase = np.zeros(labels.shape, dtype=np.float32)

    inside = labels >= 0
    phase[inside] = domain_phases[labels[inside]]

    return phase.astype(np.float32)


def make_soft_phase_from_sharp_phase(sharp_phase, support, sigma=PHASE_WALL_SIGMA):
    """
    Smooth the phase wall by smoothing the complex phasor exp(i*phase).

    This is better than directly smoothing phase values, because phase is
    circular. Direct smoothing can produce artificial jumps near -pi / pi.
    """
    support_float = support.astype(np.float32)

    real_part = np.cos(sharp_phase) * support_float
    imag_part = np.sin(sharp_phase) * support_float

    weight = gaussian_filter(support_float, sigma=sigma)
    real_smooth = gaussian_filter(real_part, sigma=sigma) / (weight + 1e-8)
    imag_smooth = gaussian_filter(imag_part, sigma=sigma) / (weight + 1e-8)

    soft_phase = np.angle(real_smooth + 1j * imag_smooth).astype(np.float32)
    soft_phase[~support] = 0.0

    return soft_phase.astype(np.float32)


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


# ============================================================
# Crystal generation
# ============================================================

def generate_irregular_crystal(seed, boundary_mode):
    """
    boundary_mode:
        "sharp" : sharp Voronoi phase domains
        "soft"  : softened phase walls between domains
    """
    if boundary_mode not in ["sharp", "soft"]:
        raise ValueError("boundary_mode must be 'sharp' or 'soft'.")

    rng = np.random.default_rng(seed)

    support, amp, support_for_plot = make_support_and_soft_amplitude(seed)

    seeds = sample_seed_points_inside_support(
        support=support,
        num_domains=NUM_DOMAINS,
        rng=rng,
    )

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

    sharp_phase = make_sharp_phase(labels, domain_phases)

    if boundary_mode == "sharp":
        phase = sharp_phase
    else:
        phase = make_soft_phase_from_sharp_phase(
            sharp_phase=sharp_phase,
            support=support,
            sigma=PHASE_WALL_SIGMA,
        )

    rho = amp.astype(np.float32) * np.exp(1j * phase.astype(np.float32))
    rho = rho.astype(np.complex64)

    return {
        "support": support,
        "support_for_plot": support_for_plot,
        "amplitude": amp,
        "phase": phase,
        "sharp_phase": sharp_phase,
        "rho": rho,
        "labels": labels,
        "seeds": seeds,
        "domain_phases": domain_phases,
        "min_adjacent_diff": min_adjacent_diff,
        "boundary_mode": boundary_mode,
    }


# ============================================================
# Projection and diffraction
# ============================================================

def project_3d_object_to_xy(crystal, eps=1e-8):
    """
    Correct coherent projection:
        rho_proj[y, x] = sum_z rho[z, y, x]

    Do not average amplitude and phase separately.
    """
    rho_3d = np.asarray(crystal["rho"], dtype=np.complex64)
    support_3d = np.asarray(crystal["support_for_plot"], dtype=bool)

    rho_proj_raw = np.sum(rho_3d, axis=0).astype(np.complex64)

    support_proj = np.any(support_3d, axis=0)

    amp_raw = np.abs(rho_proj_raw).astype(np.float32)
    amp_raw = amp_raw * support_proj.astype(np.float32)

    amp_proj = normalise_max(amp_raw)
    amp_proj = amp_proj * support_proj.astype(np.float32)

    phase_proj = np.zeros_like(amp_proj, dtype=np.float32)
    valid_phase = support_proj & (amp_raw > eps)
    phase_proj[valid_phase] = np.angle(rho_proj_raw[valid_phase]).astype(np.float32)

    rho_proj = amp_proj * np.exp(1j * phase_proj)
    rho_proj = rho_proj * support_proj.astype(np.float32)
    rho_proj = rho_proj.astype(np.complex64)

    return {
        "rho": rho_proj,
        "amp": amp_proj.astype(np.float32),
        "phase": phase_proj.astype(np.float32),
        "support": support_proj.astype(bool),
        "rho_raw": rho_proj_raw.astype(np.complex64),
        "amp_raw": amp_raw.astype(np.float32),
    }


def compute_diffraction_amplitude(rho_2d):
    dp_complex = np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(rho_2d),
            norm="ortho",
        )
    )

    dp_amp = np.abs(dp_complex).astype(np.float32)
    dp_amp = normalise_max(dp_amp)

    return dp_amp.astype(np.float32)


# ============================================================
# Main
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    crystal = generate_irregular_crystal(
        seed=SEED,
        boundary_mode="soft",
    )

    xy_projection = project_3d_object_to_xy(crystal)
    dp_amp = compute_diffraction_amplitude(xy_projection["rho"])

    amp_2d = xy_projection["amp"]
    phase_2d = xy_projection["phase"]
    support_2d = xy_projection["support"]

    save_float32_tif(OUT_DIR / "amp.tif", amp_2d)
    save_float32_tif(OUT_DIR / "phase.tif", phase_2d)
    save_float32_tif(OUT_DIR / "dp_amp.tif", dp_amp)
    save_float32_tif(OUT_DIR / "support_original.tif", support_2d.astype(np.float32))


if __name__ == "__main__":
    main()