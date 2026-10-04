# Al-Sc cooling and hold — scale-separated formulation

This example demonstrates precipitation in a spatially heterogeneous solute
field using the scale-separated monopole formulation.

A single pre-existing Al3Sc precipitate with an initial radius of 2 nm is
placed at the centre of a homogeneous Al-0.25 at.% Sc matrix. The alloy is
cooled from 550 °C to 350 °C at 5 °C/s and subsequently held at 350 °C.
During cooling, the pre-existing precipitate grows and generates a long-range
Sc-depleted region. Homogeneous nucleation remains active, allowing subsequent
precipitation to respond to the evolving spatial variation in matrix
composition.

The calculation uses a matching length of 40 nm and a 5 nm grid for the
transient long-range concentration field. Short-range particle-scale
diffusion fields are treated quasistatically, while the complementary
long-range field evolves by transient diffusion.

This example is a shortened version of the spatially heterogeneous Al-Sc
calculation described in the accompanying paper. The simulation is stopped at
a total time of 500 s to provide a more compact example.

## Files

- `material_definition.py` defines the Al-Sc material properties,
  thermodynamics, and initial matrix composition.
- `run_configuration.py` defines the simulation cell, initial particle,
  numerical controls, nucleation settings, and scale-separated transport
  parameters.
- `run.sh` defines the cooling and isothermal-hold thermal history and launches
  the calculation.
- `field_xy.json` defines a central xy plane used for concentration-field
  reconstruction.
- `reconstruct_fields.sh` reconstructs selected concentration fields from the
  saved simulation state and long-range-field snapshots.

## Running the example

With Mono'Preci installed and the appropriate Python environment activated,
run from this directory:

```bash
./run.sh
```

Alternatively, the command in `run.sh` can be executed directly using the
`monopreci` command-line interface.

The calculation uses random seed 1234. Specifying the seed makes the
stochastic nucleation sequence reproducible.

## Output

Results are written to the `results/` directory. The principal outputs are:

```text
results/
├── alsc_cool550_350_hold_seed1234.npz
├── alsc_cool550_350_hold_seed1234.dump
├── alsc_cool550_350_hold_seed1234_history.csv
└── alsc_cool550_350_hold_seed1234_lr_fields/
    ├── lr_t....npz
    └── ...
```

The main `.npz` file contains the saved simulation history and particle
states. The history CSV provides a convenient tabular summary of the
simulation. The `*_lr_fields/` directory contains snapshots of the transient
long-range concentration field.

The long-range snapshots are important for post-processing. In a
scale-separated calculation, the total matrix concentration field is

$begin:math:display$
c\(\\mathbf\{x\}\,t\)
\=
C\_\{\\rm LR\}\(\\mathbf\{x\}\,t\)
\+
\\sum\_j Q\_j\(t\)
G\_\{\\rm SR\}\(\\mathbf\{x\}\-\\mathbf\{R\}\_j\(t\)\)\.
$end:math:display$

The short-range contribution can be reconstructed from the saved particle
state, but the transient long-range field is an independently evolved state
variable. It cannot, in general, be reconstructed afterward from the particle
history alone. Therefore, simulations for which concentration-field
reconstruction is required should be run with

```python
save_hybrid_long_range_field_snapshots = True
```

as in this example.

## Reconstructing concentration fields

After the simulation has completed, selected concentration fields can be
reconstructed by running

```bash
./reconstruct_fields.sh
```

from this directory.

The supplied script reconstructs the central xy plane at selected times using
the installed `monopreci-reconstruct` command. The grid is defined by
`field_xy.json`. For this example it is a 100 × 100 cell-centred plane through
the centre of the 500 nm simulation cell, giving a spatial sampling interval
of 5 nm.

The reconstructed fields are written to

```text
results/reconstructed_fields/
```

as NumPy `.npz` files.

Each reconstructed file contains, among other metadata,

- `c` — total reconstructed concentration field,
- `c_long_range` — transient long-range contribution,
- `c_short_range` — quasistatic short-range contribution,
- `inside_precipitate` — mask identifying evaluation points inside
  precipitates,
- `particle_id` — particle associated with an interior point, where
  applicable,
- `points` — coordinates of the reconstruction grid.

Concentrations are stored as number concentrations in atoms/nm³. At points
inside precipitates, `c` is assigned the precipitate composition rather than
the continuation of the matrix diffusion field.

The example reconstruction uses CIC interpolation of the saved long-range
field:

```text
--hybrid-lr-interpolation cic
```

This is the same long-range-field sampling used by the production solver.
Spectral interpolation is also available through `monopreci-reconstruct` for
applications where a smoothly interpolated field is preferred for
visualization.

The reconstruction command can also generate VTK output for visualization in
compatible software. See

```bash
monopreci-reconstruct --help
```

for the available line, plane, volume, NPZ, VTK, time-selection, and
long-range-field options.

## Physical and numerical conditions

| Parameter | Value |
| --- | --- |
| Initial composition | Al-0.25 at.% Sc |
| Initial particle | One Al3Sc particle, R = 2 nm |
| Simulation cell | 500 × 500 × 500 nm³ |
| Initial temperature | 550 °C |
| Hold temperature | 350 °C |
| Cooling rate | 5 °C/s |
| Cooling time | 40 s |
| Isothermal hold | 460 s |
| Total simulation time | 500 s |
| Transport formulation | Scale-separated transient monopole |
| Matching length | 40 nm |
| Short-range cutoff | 120 nm |
| Long-range grid spacing | 5 nm |
| Random seed | 1234 |