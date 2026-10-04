#!/bin/bash

set -e

T_C=350
T_END_S=5000
SEED=1234

OUTDIR="results"
OUTBASE="alsc_350C_seed${SEED}"

mkdir -p "${OUTDIR}"

monopreci \
    --T-start-C "${T_C}" \
    --T-end-C "${T_C}" \
    --isothermal-time "${T_END_S}" \
    --seed "${SEED}" \
    --output "${OUTDIR}/${OUTBASE}.npz" \
    --dump-file "${OUTDIR}/${OUTBASE}.dump"
