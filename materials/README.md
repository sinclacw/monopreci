# Material definitions

This directory contains reusable material definitions for Mono'Preci.

A material definition specifies the diffusing solute species, precipitate
phases, thermodynamic parameters, diffusion data, matrix properties, and an
initial matrix composition. Calculations use a file named
`material_definition.py` in the directory from which `monopreci` is launched.

The files in this directory can therefore be used as starting points for new
calculations by copying the required definition into a calculation directory.

## Al-Sc

`material_definition_alsc.py` defines the Al-Sc / Al3Sc material system used
by the Al-Sc examples distributed with Mono'Preci and by the calculations
described in the accompanying methodology paper.

The default composition in this file is Al-0.50 at.% Sc. Individual examples
may use a different initial Sc concentration while retaining the same material
properties.

## Al-Mg-Si

`material_definition_mgsi.py` demonstrates a multisolute, multiphase material
definition containing Mg and Si solutes and beta and beta'' precipitate
phases.

This parameter set is retained primarily as an example of the more general
material interface and should not presently be regarded as a fully validated
quantitative Al-Mg-Si precipitation model. In particular, parameters
associated with the beta'' phase should be reviewed before quantitative use.

## Units

Mono'Preci material definitions use:

| Quantity | Unit |
| --- | --- |
| Length | nm |
| Diffusion prefactor | m²/s |
| Diffusion activation energy | J/mol |
| Atomic volume | nm³/atom |
| Interfacial energy | J/nm² |
| Initial composition | atomic/site fraction |

Interfacial energies commonly reported in J/m² must therefore be converted
when entered in a material definition. For example,

```python
gamma = 0.12e-18
