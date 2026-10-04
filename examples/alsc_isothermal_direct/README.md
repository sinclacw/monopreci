# Isothermal Al-Sc precipitation — direct formulation

This example demonstrates isothermal precipitation in Al-Sc using the direct
quasistatic monopole formulation.

The calculation starts from a homogeneous Al-0.50 at.% Sc matrix with no
pre-existing precipitates and is held at 350 °C for 5000 s. Al3Sc precipitates
nucleate and subsequently grow, dissolve, and coarsen according to the
particle-based precipitation model.

This type of calculation is used for the comparison between Mono'Preci and a
conventional Kampmann-Wagner numerical (KWN) model described in the
accompanying paper.

## Files

- `material_definition.py` defines the Al-Sc material properties,
  thermodynamics, and initial matrix composition.
- `run_configuration.py` defines the simulation cell, numerical controls,
  nucleation settings, and selects the direct quasistatic formulation.
- `run.sh` defines the thermal history and launches the calculation.
- `field_xy.json` defines a central xy plane for concentration-field
  reconstruction.
- `reconstruct_fields.sh` reconstructs the final concentration field on this
  plane.

## Running the example

With Mono'Preci installed and the appropriate Python environment activated,
run from this directory:

```bash
./run.sh
```

The principal output files are written to `results/`:

```text
results/
├── alsc_350C_seed1234.npz
├── alsc_350C_seed1234.dump
├── alsc_350C_seed1234_history.csv
└── alsc_350C_seed1234_nucleation_diagnostics.csv
```

The NPZ file contains the saved simulation history and particle states. The
CSV files provide global precipitation and nucleation histories, while the
dump file contains particle configurations for particle-level
post-processing.

## Concentration-field reconstruction

Because this example uses the direct quasistatic formulation,

```python
hybrid_transport = False
```

the concentration field at a saved state is determined completely by the
particle positions, radii, monopole strengths, and mean matrix composition.
No independently evolved long-range concentration field is present.

Consequently, the concentration field can be reconstructed after the
simulation directly from the saved particle state. Unlike a scale-separated
calculation, no separately saved long-range-field snapshots are required.

To reconstruct the final Sc concentration field on the central xy plane, run:

```bash
./reconstruct_fields.sh
```

This creates:

```text
results/reconstructed_fields/field_xy_final.npz
```

The supplied `field_xy.json` defines a 160 × 160 cell-centred plane through
the centre of the 80 nm cubic simulation cell:

```text
z = 40 nm
x = 0–80 nm
y = 0–80 nm
```

corresponding to a sampling interval of 0.5 nm in the x and y directions.

The reconstruction is equivalent to running:

```bash
monopreci-reconstruct \
    results/alsc_350C_seed1234.npz \
    --state final \
    --grid-file field_xy.json \
    --npz-output results/reconstructed_fields/field_xy_final.npz
```

The reconstructed concentration is stored as a number concentration in
atoms/nm³. At evaluation points lying inside Al3Sc precipitates, the reported
Sc concentration is the precipitate composition rather than a continuation of
the matrix diffusion field through the particle.

The reconstruction grid can be changed to evaluate other planes, lines, or
three-dimensional volumes. See

```bash
monopreci-reconstruct --help
```

for the available reconstruction options.

## Numerical conditions

The principal conditions for this example are:

| Parameter | Value |
|---|---:|
| Initial matrix composition | Al-0.50 at.% Sc |
| Temperature | 350 °C |
| Isothermal time | 5000 s |
| Simulation cell | 80 × 80 × 80 nm³ |
| Initial precipitates | None |
| Transport formulation | Direct quasistatic |
| Random seed | 1234 |
