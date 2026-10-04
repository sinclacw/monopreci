#!/bin/bash

TEMPERATURE_C=200
HOLD_TIME=100.0
SEED=1234

OUTBASE="v11_beta_GB_350C_seed${SEED}"

python -u -m monopole_precip.run_case \
    --T-start-C "${TEMPERATURE_C}" \
    --T-end-C "${TEMPERATURE_C}" \
    --isothermal-time "${HOLD_TIME}" \
    --seed "${SEED}" \
    --output "${OUTBASE}.npz" \
    --dump-file "${OUTBASE}.dump"
