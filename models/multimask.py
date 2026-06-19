"""Minimal triplet mask generation for ZS-SSL.

Three triplet-level modes:
- random: three independent random loss masks
- coverage: three loss masks whose union covers Omega minus Gamma
- frequencies: low-frequency, high-frequency, random

Round-robin is fixed: k = step % 3.
"""

from __future__ import annotations

import numpy as np

K = 3


def build_mask_triplet(omega_mask, rho_train, rho_val, mode, seed, gamma_mask=None):
    if mode == "random":
        return build_random_triplet(omega_mask, rho_train, rho_val, seed, gamma_mask)
    if mode == "coverage":
        return build_coverage_triplet(omega_mask, rho_train, rho_val, seed, gamma_mask)
    if mode == "frequencies":
        return build_frequency_triplet(omega_mask, rho_train, rho_val, seed, gamma_mask)
    raise ValueError("mode must be: random, coverage, or frequencies")


def build_random_triplet(omega_mask, rho_train, rho_val, seed, gamma_mask=None):
    omega, gamma, pool = _pool(omega_mask, rho_val, seed, gamma_mask)
    losses = [_sample(pool, rho_train, seed + 10 + k) for k in range(K)]
    return _pack(omega, gamma, losses)


def build_coverage_triplet(omega_mask, rho_train, rho_val, seed, gamma_mask=None):
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

    # Fill remaining loss-mask slots randomly.
    for mask_id in slots[n:]:
        available = np.flatnonzero(~selected[mask_id])
        point = int(rng.choice(available))
        selected[mask_id, point] = True

    losses = []
    for k in range(K):
        loss = np.zeros_like(pool)
        loss.ravel()[ids[selected[k]]] = 1.0
        losses.append(loss)

    return _pack(omega, gamma, losses)


def build_frequency_triplet(omega_mask, rho_train, rho_val, seed, gamma_mask=None):
    omega, gamma, pool = _pool(omega_mask, rho_val, seed, gamma_mask)
    r = _radius(pool.shape)

    losses = [
        _sample(pool, rho_train, seed + 20, np.exp(-((r / 0.35) ** 2))),
        _sample(pool, rho_train, seed + 21, r**2 + 1e-6),
        _sample(pool, rho_train, seed + 22),
    ]

    return _pack(omega, gamma, losses)


def add_multimask_args(parser):
    parser.add_argument("--use_multi_masks", action="store_true")
    parser.add_argument(
        "--multi_mask_mode",
        required=True,
        choices=("random", "coverage", "frequencies"),
    )
    parser.add_argument("--multi_mask_seed", type=int, default=0)
    return parser


def select_round_robin(bank, step):
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


def _pack(omega, gamma, losses):
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
    out = np.zeros_like(mask, dtype=np.float32)

    if n == 0:
        return out

    probs = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).ravel()[ids]
        w = np.maximum(w, 0.0)
        probs = None if w.sum() == 0 else w / w.sum()

    rng = np.random.default_rng(seed)
    out.ravel()[rng.choice(ids, size=n, replace=False, p=probs)] = 1.0
    return out


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
