#!/bin/bash

T_START_C=550
T_END_C=200
COOLING_RATE=1.0
TEMPERATURE_STEP=1.0
SEED=1235

OUTBASE="v11_beta_cooling_seed${SEED}"

python -u -m monopole_precip.run_case \
    --T-start-C "${T_START_C}" \
    --T-end-C "${T_END_C}" \
    --cooling-rate "${COOLING_RATE}" \
    --temperature-step "${TEMPERATURE_STEP}" \
    --seed "${SEED}" \
    --output "${OUTBASE}.npz" \
    --dump-file "${OUTBASE}.dump"

