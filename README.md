# Mono'Preci

Mono'Preci is a particle-based precipitation-kinetics code for modelling
nucleation, growth, dissolution, and coarsening in spatially heterogeneous
solute fields. Precipitates are represented explicitly as spherical particles,
while solute transport is described using monopole Green's-function solutions.

The scale-separated formulation combines quasistatic particle-scale diffusion
fields with a transient long-range concentration field. An Ewald decomposition
separates the concentration field by spatial scale: screened short-range
monopole interactions describe the local diffusion fields surrounding
individual precipitates, while the complementary long-range field is evolved
by transient diffusion on a periodic grid.

The code supports multicomponent alloy systems, multiple precipitate phases,
stochastic classical nucleation theory (CNT), continuous thermal histories,
and spatially resolved precipitation kinetics.

Mono'Preci accompanies the methodology described in:

> C. W. Sinclair, *A Scale-Separated Full-Field Monopole Method for
> Precipitation Modelling*, publication details to be added.

## Installation

Python 3.12 is recommended.

Clone the repository and install the package in an activated Python
environment:

```bash
git clone https://github.com/sinclacw/monopreci.git
cd monopreci
python -m pip install -e .
```

For example, using conda:

```bash
conda create -n monopreci python=3.12
conda activate monopreci

git clone https://github.com/sinclacw/monopreci.git
cd monopreci
python -m pip install -e .
```

On systems where conda is unavailable, a standard Python virtual environment
can be used:

```bash
python -m venv ~/venvs/monopreci
source ~/venvs/monopreci/bin/activate
python -m pip install --upgrade pip

git clone https://github.com/sinclacw/monopreci.git
cd monopreci
python -m pip install -e .
```

Installation provides two command-line entry points:

```bash
monopreci --help
monopreci-reconstruct --help
```

`monopreci` runs precipitation simulations, while `monopreci-reconstruct`
reconstructs spatial concentration fields from saved simulation states.

## Quick start

Two Al-Sc examples are included.

### Isothermal precipitation

The first example uses the direct quasistatic monopole formulation to simulate
precipitation from an initially homogeneous Al-0.50 at.% Sc matrix at 350 °C:

```bash
cd examples/alsc_isothermal_direct
./run.sh
```

This calculation demonstrates stochastic nucleation followed by precipitate
growth, dissolution, and coarsening.

### Cooling and isothermal hold

The second example uses the scale-separated formulation:

```bash
cd examples/alsc_cool_hold_scale_separated
./run.sh
```

A pre-existing Al3Sc precipitate grows while an Al-0.25 at.% Sc alloy is
cooled from 550 °C to 350 °C at 5 °C/s and subsequently held at 350 °C.
Growth of the initial particle produces a long-range Sc-depleted region that
influences subsequent nucleation.

This example also demonstrates reconstruction of the spatial concentration
field from saved particle states and transient long-range-field snapshots:

```bash
./reconstruct_fields.sh
```

See the `README.md` within each example directory for the complete physical,
numerical, and post-processing conditions.

## Configuring a calculation

A Mono'Preci calculation is defined primarily by two Python files in the
directory from which the calculation is launched:

```text
material_definition.py
run_configuration.py
```

The thermal history, random seed, and output filenames are supplied through
the `monopreci` command-line interface.

### Material definition

`material_definition.py` defines the alloy and precipitate phases, including:

- solute diffusivities;
- matrix and precipitate atomic volumes;
- precipitate stoichiometry;
- interfacial energy;
- thermodynamic model parameters;
- initial matrix composition.

The examples provide a complete material definition for Al-Sc.

### Run configuration

`run_configuration.py` defines the spatial and numerical configuration,
including:

- simulation-cell dimensions;
- initial particle positions and radii;
- adaptive timestep controls;
- nucleation controls;
- particle-removal criteria;
- optional grain-boundary geometry;
- choice of transport formulation;
- scale-separation and long-range-grid parameters.

This separation allows the same installed Mono'Preci package to be used for
different calculations without copying or modifying the solver source code.

## Transport formulations

Mono'Preci provides two related descriptions of solute transport.

### Direct quasistatic formulation

In the direct formulation, the concentration field is represented by the
periodic monopole Green's function and is assumed to relax quasistatically.
This provides a particle-based description of interacting precipitate growth
without resolving the surrounding diffusion field on a spatial mesh.

The direct formulation is selected with:

```python
hybrid_transport = False
```

It is appropriate when the relevant diffusion fields relax rapidly compared
with the timescale of precipitate evolution.

### Scale-separated formulation

For systems containing concentration variations over substantially larger
length scales, the concentration field is decomposed into short- and
long-range contributions,

$begin:math:display$
c\(\\mathbf\{x\}\,t\)
\=
C\_\{\\mathrm\{LR\}\}\(\\mathbf\{x\}\,t\)
\+
\\sum\_j Q\_j
G\_\{\\mathrm\{SR\}\}\(\\mathbf\{x\}\-\\mathbf\{R\}\_j\;\\ell\)\.
$end:math:display$

The screened short-range Green's function is

$begin:math:display$
G\_\{\\mathrm\{SR\}\}\(r\;\\ell\)
\=
\\frac\{\\operatorname\{erfc\}\(r\/\\ell\)\}\{4\\pi r\}
\-
\\frac\{\\ell\^2\}\{4V\}\.
$end:math:display$

The particle-scale short-range fields are treated quasistatically, while the
complementary long-range field evolves according to transient diffusion.
Growth and dissolution of the discrete precipitates provide conservative
sources and sinks to the long-range field.

The scale-separated formulation is selected with:

```python
hybrid_transport = True
```

The principal transport controls are:

```python
# Matching length ell [nm].
hybrid_matching_length_nm = 40.0

# Short-range cutoff expressed as a multiple of ell.
hybrid_short_cutoff_factor = 3.0

# Target spacing of the long-range grid [nm].
hybrid_grid_spacing_nm = 5.0
```

The matching length should be chosen so that particle-scale diffusion relaxes
rapidly relative to particle evolution. Predictions should be checked for
sensitivity to its precise value.

The short-range cutoff and long-range grid spacing control the numerical
resolution of the scale-separated representation. The long-range grid may be
rectangular, with the number of grid points in each direction determined from
the simulation-cell dimensions and target grid spacing.

## Running calculations

The command line specifies the thermal history, random realization, and output
files. Model and numerical parameters remain in `material_definition.py` and
`run_configuration.py`.

An isothermal calculation can be launched with:

```bash
monopreci \
    --T-start-C 350 \
    --T-end-C 350 \
    --isothermal-time 5000 \
    --seed 1234 \
    --output results.npz \
    --dump-file particles.dump
```

A continuous cooling calculation uses:

```bash
monopreci \
    --T-start-C 550 \
    --T-end-C 350 \
    --cooling-rate 5 \
    --temperature-step 1 \
    --seed 1234 \
    --output results.npz \
    --dump-file particles.dump
```

Cooling can be followed immediately by an isothermal hold by additionally
specifying:

```bash
--isothermal-time 460
```

The cooling and hold form one continuous calculation. The particle population
and, for scale-separated calculations, the long-range concentration field are
carried continuously between thermal segments.

## Output

The principal outputs are an NPZ results file and a particle dump file.

The NPZ file contains simulation histories and run metadata, including
quantities such as:

- time and temperature;
- particle radii and positions;
- particle number;
- matrix composition;
- precipitate volume fraction;
- monopole strengths;
- transport and numerical diagnostics.

Because nucleation and particle extinction change the number of particles,
some particle histories are stored as variable-length data structures.

The dump file contains particle configurations at selected times and can be
used for particle-level post-processing and visualization.

Scale-separated calculations can additionally save snapshots of the transient
long-range concentration field. These are enabled in `run_configuration.py`
with:

```python
save_hybrid_long_range_field_snapshots = True
```

Saving these snapshots is required if the spatial concentration field of a
scale-separated simulation will be reconstructed after the calculation.

## Concentration-field reconstruction

Mono'Preci provides the `monopreci-reconstruct` command for reconstructing
spatial concentration fields on user-defined line, plane, or volume grids.

For the direct quasistatic formulation, the complete matrix concentration
field can be reconstructed from the saved particle state.

For the scale-separated formulation,

$begin:math:display$
c\(\\mathbf\{x\}\,t\)
\=
C\_\{\\mathrm\{LR\}\}\(\\mathbf\{x\}\,t\)
\+
c\_\{\\mathrm\{SR\}\}\(\\mathbf\{x\}\,t\)\,
$end:math:display$

the short-range contribution can be reconstructed from the particle state,
but the transient long-range field is an independently evolved state
variable. It cannot, in general, be reconstructed afterward from the particle
history alone.

Therefore, if post-run concentration-field reconstruction is required for a
scale-separated calculation, long-range snapshots must be enabled during the
simulation:

```python
save_hybrid_long_range_field_snapshots = True
```

A reconstruction requires a saved simulation state, a grid definition, and,
for a scale-separated calculation, the corresponding long-range-field
snapshot. For example:

```bash
monopreci-reconstruct \
    results.npz \
    --time 490 \
    --time-selection exact \
    --grid-file field_xy.json \
    --lr-dir results_lr_fields \
    --hybrid-lr-interpolation cic \
    --npz-output field_xy_490s.npz
```

The grid definition is a JSON file describing a line, plane, or volume. The
scale-separated Al-Sc example contains a complete plane-reconstruction
workflow.

Reconstructed NPZ files contain the total field together with its
scale-separated components:

```text
c
c_long_range
c_short_range
```

They also contain the reconstruction-grid coordinates, an
`inside_precipitate` mask, particle identifiers where applicable, and
simulation metadata.

Concentration fields are stored as number concentrations in atoms/nm³. At
evaluation points inside precipitates, the total concentration is assigned the
corresponding precipitate composition rather than the mathematical
continuation of the matrix diffusion field.

For scale-separated calculations, CIC interpolation reproduces the
long-range-field sampling used by the production solver:

```text
--hybrid-lr-interpolation cic
```

Spectral interpolation is also available for applications where a smoothly
interpolated long-range field is preferred for visualization.

The reconstructed field can be written as NumPy NPZ data, VTK data, or both.
See

```bash
monopreci-reconstruct --help
```

for the complete set of reconstruction options.

## Numerical considerations

Mono'Preci uses adaptive time integration. The principal timestep controls in
`run_configuration.py` are:

```python
dt_initial
first_step_dt_max
dt_max
dt_min
dt_safety
isothermal_segment_time_s
```

The internal timestep is constrained by the evolving precipitation kinetics
and numerical safety criteria. `isothermal_segment_time_s` controls the
segmentation of the imposed thermal history rather than the internal adaptive
integration timestep.

For scale-separated calculations, the long-range diffusion equation is
advanced spectrally on the periodic grid. The spectral diffusion update is not
subject to an explicit diffusion CFL stability limit, although the grid must
be sufficiently fine to resolve the long-range concentration field.

## Model assumptions

The present formulation assumes:

- spherical precipitates;
- diffusion-controlled solute transport;
- a periodic simulation domain;
- precipitate compositions defined by the selected phase model;
- no elastic interaction energy in the spatial particle coupling.

An optional planar grain-boundary region can be defined for heterogeneous
nucleation calculations.

## Repository structure

The principal directories are:

```text
monopreci/
├── examples/
│   ├── alsc_isothermal_direct/
│   └── alsc_cool_hold_scale_separated/
├── materials/
├── src/
│   └── monopole_precip/
├── README.md
└── pyproject.toml
```

The main solver components under `src/monopole_precip/` include:

```text
run_case.py
    Command-line simulation driver.

thermal_profile.py
    Construction of cooling, isothermal, and cooling-plus-hold histories.

ternary_thermal_history.py
    Evolution through successive thermal-history segments.

integrator.py
    Adaptive particle-growth and nucleation integration.

ternary_growth_coupled.py
    Coupled particle/interface growth solution.

hybrid_short_range.py
    Short-range monopole operator for the scale-separated formulation.

hybrid_long_range.py
    Transient periodic long-range diffusion field.

nucleation_growth_subfunctions.py
    Nucleation, growth, and associated timestep calculations.

outputs_subfunctions.py
    Simulation-output utilities.

concentrationfield/
    Concentration-field reconstruction and VTK/NumPy output utilities.
```

## Reproducibility

The example calculations specify their random seeds explicitly. Stochastic
nucleation calculations can therefore be repeated using the same material
definition, run configuration, thermal history, and seed.

The included example calculations also provide compact end-to-end checks of
the direct and scale-separated formulations.

## Citation

If you use Mono'Preci in published work, please cite:

> C. W. Sinclair, *A Scale-Separated Full-Field Monopole Method for
> Precipitation Modelling*, publication details to be added.

Citation information will be updated when the accompanying paper is
published.

## License

Copyright © 2026 The University of British Columbia.

Mono'Preci is made available under the UBC Research Use
Software License.

The software may be used, modified, and redistributed for
research, teaching, and educational purposes by individuals
and organizations, including academic, nonprofit, government,
and for-profit organizations, subject to the terms of the license.

Permitted activities include internal research and development,
evaluation, and testing. Commercial use, as defined in the
license, is not permitted without a separate agreement.

See [LICENSE](LICENSE) for the complete terms and conditions.

For commercial licensing inquiries, contact
techtransfer@innovation.ubc.ca.

## Disclaimer

Mono'Preci is provided "AS IS", without warranty of any kind.

The software is distributed under the UBC Research Use Software
License. Please refer to the [LICENSE](LICENSE) file for the
complete warranty disclaimer, limitations of liability, and
conditions of use.

## Citation

If you use Mono'Preci in published research, please cite:

C. W. Sinclair,
*A Scale-Separated Full-Field Monopole Method for
Precipitation Modelling*,
arXiv:2610.05436 (2026).
DOI: 10.48550/arXiv.2610.05436.

Publications arising from use of Mono'Preci must also include
the acknowledgment specified in Section 9 of the license:

> The Software used in this research was created by
> Chad Sinclair of The University of British Columbia
> (“UBC”). © 2026 UBC.
