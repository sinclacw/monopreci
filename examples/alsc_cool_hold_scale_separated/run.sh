#!/bin/bash

set -e

T_START_C=550
T_END_C=350
COOLING_RATE=5
TEMPERATURE_STEP=1
ISOTHERMAL_TIME=460
SEED=1234

OUTDIR="results"
OUTBASE="alsc_cool550_350_hold_seed${SEED}"

mkdir -p "${OUTDIR}"

monopreci \
    --T-start-C "${T_START_C}" \
    --T-end-C "${T_END_C}" \
    --cooling-rate "${COOLING_RATE}" \
    --temperature-step "${TEMPERATURE_STEP}" \
    --isothermal-time "${ISOTHERMAL_TIME}" \
    --seed "${SEED}" \
    --output "${OUTDIR}/${OUTBASE}.npz" \
    --dump-file "${OUTDIR}/${OUTBASE}.dump"
