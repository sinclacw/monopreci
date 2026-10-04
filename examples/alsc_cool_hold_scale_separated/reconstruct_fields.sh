#!/bin/bash
set -e

# Reconstruct the Sc concentration field on the central xy plane.
#
# Scale-separated simulations require the saved long-range field
# snapshots in addition to the particle-state history.

RESULTS="results"
SIMULATION="${RESULTS}/alsc_cool550_350_hold_seed1234.npz"
LR_DIR="${RESULTS}/alsc_cool550_350_hold_seed1234_lr_fields"
GRID="field_xy.json"
OUTDIR="${RESULTS}/reconstructed_fields"

mkdir -p "${OUTDIR}"

# Selected times illustrate:
#   40 s  : end of cooling
#   240 s : intermediate hold
#   490 s : late hold
#
# These times correspond to saved simulation states and LR snapshots.

for TIME in 40 240 490
do
    echo "Reconstructing field at t = ${TIME} s"

    monopreci-reconstruct \
        "${SIMULATION}" \
        --time "${TIME}" \
        --time-selection exact \
        --grid-file "${GRID}" \
        --lr-dir "${LR_DIR}" \
        --hybrid-lr-interpolation cic \
        --npz-output "${OUTDIR}/field_xy_${TIME}s.npz"
done
