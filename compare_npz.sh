#!/bin/bash

set -euo pipefail

ANALYTIC="${1:-results/smoke_analytic_350C_seed1234.npz}"
BASELINE="${2:-../v8-baseline/results/smoke_baseline_350C_seed1234.npz}"

python - "$ANALYTIC" "$BASELINE" <<'PY'
import sys
import numpy as np

file_a = sys.argv[1]
file_b = sys.argv[2]

a = np.load(file_a, allow_pickle=True)
b = np.load(file_b, allow_pickle=True)

print("=" * 78)
print("NPZ comparison")
print("=" * 78)
print("Analytic :", file_a)
print("Baseline :", file_b)
print()

keys_a = set(a.files)
keys_b = set(b.files)

only_a = sorted(keys_a - keys_b)
only_b = sorted(keys_b - keys_a)
common = sorted(keys_a & keys_b)

if only_a:
    print("Keys only in analytic:")
    for k in only_a:
        print("  ", k)
    print()

if only_b:
    print("Keys only in baseline:")
    for k in only_b:
        print("  ", k)
    print()

print(f"Common keys: {len(common)}")
print()

def compare_numeric(name, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        print(
            f"{name:30s} "
            f"SHAPE MISMATCH  {x.shape} vs {y.shape}"
        )
        return False

    if x.size == 0:
        print(f"{name:30s} empty")
        return True

    diff = np.abs(x - y)

    max_abs = float(np.max(diff))

    scale = np.maximum(
        np.maximum(np.abs(x), np.abs(y)),
        1.0e-300,
    )

    max_rel = float(
        np.max(diff / scale)
    )

    all_equal = np.array_equal(x, y)

    print(
        f"{name:30s} "
        f"shape={str(x.shape):14s} "
        f"exact={str(all_equal):5s} "
        f"max_abs={max_abs:.6e} "
        f"max_rel={max_rel:.6e}"
    )

    return all_equal


all_exact = True

for key in common:

    x = a[key]
    y = b[key]

    # ----------------------------------------------------------
    # Ordinary numeric arrays
    # ----------------------------------------------------------
    if (
        x.dtype != object
        and y.dtype != object
        and np.issubdtype(x.dtype, np.number)
        and np.issubdtype(y.dtype, np.number)
    ):
        exact = compare_numeric(key, x, y)
        all_exact = all_exact and exact
        continue

    # ----------------------------------------------------------
    # Object arrays, e.g. variable-length histories such as
    # R_hist, Q_mg_hist, Q_si_hist, dRdt_hist.
    # ----------------------------------------------------------
    if x.dtype == object or y.dtype == object:

        if x.shape != y.shape:
            print(
                f"{key:30s} "
                f"SHAPE MISMATCH  {x.shape} vs {y.shape}"
            )
            all_exact = False
            continue

        max_abs = 0.0
        max_rel = 0.0
        exact = True
        mismatch_index = None

        for i, (xi, yi) in enumerate(zip(x.flat, y.flat)):

            try:
                xi = np.asarray(xi, dtype=float)
                yi = np.asarray(yi, dtype=float)
            except Exception:
                if str(xi) != str(yi):
                    exact = False
                    if mismatch_index is None:
                        mismatch_index = i
                continue

            if xi.shape != yi.shape:
                exact = False
                if mismatch_index is None:
                    mismatch_index = i
                continue

            if xi.size == 0:
                continue

            d = np.abs(xi - yi)

            this_abs = float(np.max(d))

            scale = np.maximum(
                np.maximum(
                    np.abs(xi),
                    np.abs(yi),
                ),
                1.0e-300,
            )

            this_rel = float(
                np.max(d / scale)
            )

            max_abs = max(max_abs, this_abs)
            max_rel = max(max_rel, this_rel)

            if not np.array_equal(xi, yi):
                exact = False
                if mismatch_index is None:
                    mismatch_index = i

        msg = (
            f"{key:30s} "
            f"shape={str(x.shape):14s} "
            f"exact={str(exact):5s} "
            f"max_abs={max_abs:.6e} "
            f"max_rel={max_rel:.6e}"
        )

        if mismatch_index is not None:
            msg += f" first_diff={mismatch_index}"

        print(msg)

        all_exact = all_exact and exact
        continue

    # ----------------------------------------------------------
    # Non-numeric metadata
    # ----------------------------------------------------------
    exact = np.array_equal(x, y)

    print(
        f"{key:30s} "
        f"exact={str(exact):5s} "
        f"dtype={x.dtype}"
    )

    all_exact = all_exact and exact


print()
print("=" * 78)

if all_exact:
    print("RESULT: files are exactly identical for all common stored values.")
else:
    print(
        "RESULT: differences exist. "
        "Inspect max_abs/max_rel values above."
    )

print("=" * 78)
PY
