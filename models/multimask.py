"""Minimal triplet mask generation for ZS-SSL.

Three triplet-level modes:
- random: three independent random loss masks
- coverage: three loss masks whose union covers Omega minus Gamma
- frequencies: low-frequency, high-frequency, random

Round-robin is fixed: k = step % 3.
"""

from __future__ import annotations

import numpy as np

def build_mask_triplet(omega_mask, rho_train, rho_val, mode, seed, K=3, gamma_mask=None):
    if mode == "random":
        return build_random_triplet(omega_mask, rho_train, rho_val, seed, K, gamma_mask)
    if mode == "coverage":
        return build_coverage_triplet(omega_mask, rho_train, rho_val, seed, K, gamma_mask)
    if mode == "frequencies":
        return build_frequency_triplet(omega_mask, rho_train, rho_val, seed, K, gamma_mask)
    if mode == "frequency_balanced":
        return build_frequency_balanced_triplet(omega_mask, rho_train, rho_val, seed, K, gamma_mask)
    if mode == "frequency_curriculum":
        return build_frequency_curriculum_triplet(omega_mask, rho_train, rho_val, seed, K, gamma_mask)
    raise ValueError("mode must be: random, coverage, frequencies, frequency_balanced, or frequency_curriculum")


def build_random_triplet(omega_mask, rho_train, rho_val, seed, K, gamma_mask=None):
    omega, gamma, pool = _pool(omega_mask, rho_val, seed, gamma_mask)
    losses = [_sample(pool, rho_train, seed + 10 + k) for k in range(K)]
    return _pack(omega, gamma, losses, K)


def build_coverage_triplet(omega_mask, rho_train, rho_val, seed, K, gamma_mask=None):
    omega, gamma, pool = _pool(omega_mask, rho_val, seed, gamma_mask)

    ids = np.flatnonzero(pool.ravel())
    n = len(ids)
    n_each = _count(rho_train, n)

    if n == 0:
        raise ValueError("No non-validation acquired samples.")
    if K * n_each < n:
        raise ValueError(f"coverage needs rho_train >= about 1/{K}")

    rng = np.random.default_rng(seed + 100)
    selected = np.zeros((K, n), dtype=bool)

    slots = np.repeat(np.arange(K), n_each)
    rng.shuffle(slots)

    # First pass: every acquired non-validation index appears in one loss mask.
    for point, mask_id in zip(rng.permutation(n), slots[:n]):
        selected[mask_id, point] = True

    # Fill remaining loss-mask slots randomly (Vectorized for high K performance)
    mask_ids, counts = np.unique(slots[n:], return_counts=True)
    for mask_id, count in zip(mask_ids, counts):
        available = np.flatnonzero(~selected[mask_id])
        chosen = rng.choice(available, size=count, replace=False)
        selected[mask_id, chosen] = True

    losses = []
    for k in range(K):
        loss = np.zeros(pool.size, dtype=np.float32)
        loss[ids[selected[k]]] = 1.0
        losses.append(loss.reshape(pool.shape))

    return _pack(omega, gamma, losses, K)


def build_frequency_triplet(omega_mask, rho_train, rho_val, seed, K, gamma_mask=None):
    omega, gamma, pool = _pool(omega_mask, rho_val, seed, gamma_mask)
    r = _radius(pool.shape)

    losses = [
        _sample(pool, rho_train, seed + 20, np.exp(-((r / 0.35) ** 2))),
    ]
    if K > 1:
        losses.append(_sample(pool, rho_train, seed + 21, r**2 + 1e-6))
    
    # Fill remaining slots with random distributions
    for k in range(2, K):
        losses.append(_sample(pool, rho_train, seed + 22 + k))

    return _pack(omega, gamma, losses, K)


def build_frequency_balanced_triplet(omega_mask, rho_train, rho_val, seed, K, gamma_mask=None):
    omega, gamma, pool = _pool(omega_mask, rho_val, seed, gamma_mask)
    r = _radius(pool.shape)

    low_thr = 0.33
    high_thr = 0.66
    band_fractions = (0.15, 0.45, 0.40)

    # Convert pool to boolean array for logical AND
    pool_bool = pool > 0
    low_pool = pool_bool & (r <= low_thr)
    mid_pool = pool_bool & (r > low_thr) & (r <= high_thr)
    high_pool = pool_bool & (r > high_thr)

    ids_low = np.flatnonzero(low_pool.ravel())
    ids_mid = np.flatnonzero(mid_pool.ravel())
    ids_high = np.flatnonzero(high_pool.ravel())

    n_total_loss = _count(rho_train, np.count_nonzero(pool_bool))

    f_low, f_mid, f_high = band_fractions
    s = f_low + f_mid + f_high
    f_low, f_mid, f_high = f_low / s, f_mid / s, f_high / s

    n_low = int(round(f_low * n_total_loss))
    n_mid = int(round(f_mid * n_total_loss))
    n_high = n_total_loss - n_low - n_mid

    n_low = min(n_low, len(ids_low))
    n_mid = min(n_mid, len(ids_mid))
    n_high = min(n_high, len(ids_high))

    losses = []
    for k in range(K):
        rng = np.random.default_rng(seed + 100 + k)
        loss = np.zeros(pool.size, dtype=np.float32)
        if n_low > 0:
            loss[rng.choice(ids_low, size=n_low, replace=False)] = 1.0
        if n_mid > 0:
            loss[rng.choice(ids_mid, size=n_mid, replace=False)] = 1.0
        if n_high > 0:
            loss[rng.choice(ids_high, size=n_high, replace=False)] = 1.0
        losses.append(loss.reshape(pool.shape))

    return _pack(omega, gamma, losses, K)


def add_multimask_args(parser):
    parser.add_argument("--use_multi_masks", action="store_true")
    parser.add_argument(
        "--multi_mask_mode",
        required=True,
        choices=("random", "coverage", "frequencies", "frequency_balanced"),
    )
    parser.add_argument("--multi_mask_seed", type=int, default=0)
    return parser


def select_round_robin(bank, step, K):
    k = step % K
    return bank["trn_masks"][k], bank["loss_masks"][k], k


def _pool(omega_mask, rho_val, seed, gamma_mask):
    omega = _binary(omega_mask)

    if gamma_mask is None:
        gamma = _sample(omega, rho_val, seed)
    else:
        gamma = _binary(gamma_mask)
        if gamma.shape != omega.shape:
            raise ValueError("gamma_mask and omega_mask must have the same shape")
        if np.any(gamma > omega):
            raise ValueError("gamma_mask must be a subset of omega_mask")

    return omega, gamma, omega - gamma


def _pack(omega, gamma, losses, K):
    loss_masks = np.stack(losses).astype(np.float32)
    trn_masks = np.stack([omega - gamma - loss for loss in losses]).astype(np.float32)

    for k in range(K):
        assert np.all(gamma * loss_masks[k] == 0)
        assert np.all(gamma * trn_masks[k] == 0)
        assert np.all(loss_masks[k] * trn_masks[k] == 0)
        np.testing.assert_array_equal(gamma + loss_masks[k] + trn_masks[k], omega)

    return {
        "trn_masks": trn_masks,
        "loss_masks": loss_masks,
        "gamma_mask": gamma.astype(np.float32),
        "omega_mask": omega.astype(np.float32),
    }


def _sample(mask, rho, seed, weights=None):
    ids = np.flatnonzero(mask.ravel())
    n = _count(rho, len(ids))
    out = np.zeros(mask.size, dtype=np.float32)

    if n == 0:
        return out.reshape(mask.shape)

    probs = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).ravel()[ids]
        w = np.maximum(w, 0.0)
        probs = None if w.sum() == 0 else w / w.sum()

    rng = np.random.default_rng(seed)
    out[rng.choice(ids, size=n, replace=False, p=probs)] = 1.0
    return out.reshape(mask.shape)


def _count(rho, total):
    if not 0 <= rho <= 1:
        raise ValueError("rho must be in [0, 1]")
    return int(round(rho * total))


def _binary(mask):
    return (np.asarray(mask) > 0).astype(np.float32)


def _radius(shape):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - h // 2) ** 2 + (x - w // 2) ** 2)
    return r / (r.max() + 1e-12)

def curriculum_band_fractions(k, K=25):
    """
    Frequency curriculum for the LOSS mask Lambda_k.
    k = 0 starts low/mid focused.
    k = K-1 ends mildly high-frequency focused.
    """
    t = k / max(K - 1, 1)

    start = np.array([0.32, 0.45, 0.23], dtype=np.float32)
    end   = np.array([0.25, 0.45, 0.30], dtype=np.float32)


    frac = (1.0 - t) * start + t * end
    frac = frac / frac.sum()

    return float(frac[0]), float(frac[1]), float(frac[2])

def build_frequency_curriculum_triplet(omega_mask, rho_train, rho_val, seed, K, gamma_mask=None):
    omega, gamma, pool = _pool(omega_mask, rho_val, seed, gamma_mask)
    
    r = _radius(omega.shape)
    ids = np.flatnonzero(pool.ravel())
    r_pool = r.ravel()[ids]
    
    # 15% low, 40% mid, 45% high
    low_thresh = np.percentile(r_pool, 15)
    mid_thresh = np.percentile(r_pool, 55)
    
    low_mask = (r <= low_thresh) & (pool > 0)
    mid_mask = (r > low_thresh) & (r <= mid_thresh) & (pool > 0)
    high_mask = (r > mid_thresh) & (pool > 0)
    
    n_total = _count(rho_train, len(ids))
    losses = []
    
    rng = np.random.default_rng(seed)
    
    for k in range(K):
        low_frac, mid_frac, high_frac = curriculum_band_fractions(k, K)
        
        n_low = int(round(low_frac * n_total))
        n_mid = int(round(mid_frac * n_total))
        n_high = n_total - n_low - n_mid
        
        loss_k = np.zeros(pool.size, dtype=np.float32)
        loss_k[rng.choice(np.flatnonzero(low_mask), size=n_low, replace=False)] = 1.0
        loss_k[rng.choice(np.flatnonzero(mid_mask), size=n_mid, replace=False)] = 1.0
        loss_k[rng.choice(np.flatnonzero(high_mask), size=n_high, replace=False)] = 1.0
        
        losses.append(loss_k.reshape(pool.shape))
        
    return _pack(omega, gamma, losses, K)
