"""CPU tests for models/multimask.py."""

from __future__ import annotations

import argparse

import numpy as np

try:
    from models.multimask import (
        add_multimask_args,
        build_coverage_triplet,
        build_frequency_triplet,
        build_mask_triplet,
        build_random_triplet,
        select_round_robin,
    )
except ImportError:
    from multimask_minimal import (
        add_multimask_args,
        build_coverage_triplet,
        build_frequency_triplet,
        build_mask_triplet,
        build_random_triplet,
        select_round_robin,
    )


def omega_mask(shape=(48, 48), acceleration=4, center_width=8):
    mask = np.zeros(shape, dtype=np.float32)
    mask[:, ::acceleration] = 1.0
    c = shape[1] // 2
    mask[:, c - center_width // 2 : c + center_width // 2] = 1.0
    return mask


def check_partition(bank):
    omega = bank["omega_mask"]
    gamma = bank["gamma_mask"]
    trn = bank["trn_masks"]
    loss = bank["loss_masks"]

    assert trn.shape == loss.shape == (3, *omega.shape)
    assert gamma.shape == omega.shape

    for k in range(3):
        assert np.all(gamma * trn[k] == 0)
        assert np.all(gamma * loss[k] == 0)
        assert np.all(trn[k] * loss[k] == 0)
        np.testing.assert_array_equal(gamma + trn[k] + loss[k], omega)


def radial_mean(mask):
    h, w = mask.shape
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - h // 2) ** 2 + (x - w // 2) ** 2)
    r = r / (r.max() + 1e-12)
    return float(r[mask.astype(bool)].mean())


def test_random_triplet():
    bank = build_random_triplet(omega_mask(), rho_train=0.4, rho_val=0.1, seed=0)
    check_partition(bank)
    assert not np.array_equal(bank["loss_masks"][0], bank["loss_masks"][1])


def test_coverage_triplet():
    omega = omega_mask(shape=(32, 32), acceleration=2, center_width=0)
    bank = build_coverage_triplet(omega, rho_train=0.4, rho_val=0.0, seed=1)
    check_partition(bank)

    union = (bank["loss_masks"].sum(axis=0) > 0).astype(np.float32)
    np.testing.assert_array_equal(union, bank["omega_mask"])


def test_frequency_triplet():
    omega = np.ones((96, 96), dtype=np.float32)
    bank = build_frequency_triplet(omega, rho_train=0.25, rho_val=0.0, seed=2)
    check_partition(bank)

    low = radial_mean(bank["loss_masks"][0])
    high = radial_mean(bank["loss_masks"][1])
    random = radial_mean(bank["loss_masks"][2])
    assert low < random < high, (low, random, high)


def test_dispatcher_and_round_robin():
    bank = build_mask_triplet(
        omega_mask(), rho_train=0.4, rho_val=0.0, mode="random", seed=3
    )

    used = []
    for step in range(7):
        trn, loss, k = select_round_robin(bank, step)
        used.append(k)
        np.testing.assert_array_equal(trn, bank["trn_masks"][k])
        np.testing.assert_array_equal(loss, bank["loss_masks"][k])

    assert used == [0, 1, 2, 0, 1, 2, 0]


def test_parser_requires_triplet_mode():
    parser = add_multimask_args(argparse.ArgumentParser())
    args = parser.parse_args(
        ["--use_multi_masks", "--multi_mask_mode", "coverage", "--multi_mask_seed", "5"]
    )

    assert args.use_multi_masks
    assert args.multi_mask_mode == "coverage"
    assert args.multi_mask_seed == 5
