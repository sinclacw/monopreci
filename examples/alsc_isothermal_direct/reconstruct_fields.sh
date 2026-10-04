#!/bin/bash
set -e

# Reconstruct the Sc concentration field on the central xy plane.
#
# For the direct quasistatic formulation, the complete concentration
# field can be reconstructed from the saved particle state. No separately
# saved long-range field is required.

RESULTS="results"
SIMULATION="${RESULTS}/alsc_350C_seed1234.npz"
GRID="field_xy.json"
OUTDIR="${RESULTS}/reconstructed_fields"

mkdir -p "${OUTDIR}"

echo "Reconstructing final concentration field"

monopreci-reconstruct \
    "${SIMULATION}" \
    --state final \
    --grid-file "${GRID}" \
    --npz-output "${OUTDIR}/field_xy_final.npz"
