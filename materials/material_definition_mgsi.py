"""
Al-Mg-Si material definition for Mono'Preci.

This file is retained as an example of a multisolute, multiphase material
definition. It defines Mg and Si as diffusing solutes and includes beta and
beta'' precipitate phases.

The parameter set should be regarded as provisional rather than as a
validated quantitative Al-Mg-Si precipitation model. In particular, the
parameters associated with the beta'' phase should be reviewed before this
definition is used for quantitative predictions.

Units
-----
Diffusion prefactor : m^2/s
Activation energy   : J/mol
Atomic volume       : nm^3/atom
Interfacial energy  : J/nm^2
Lattice parameter   : nm

Initial compositions are atomic/site fractions.
"""

from monopole_precip.phases import (
    AlloySystem,
    PrecipitatePhase,
    SolubilityProductThermodynamics,
    SoluteSpecies,
)


# ============================================================================
# Diffusing solute species
# ============================================================================

Mg = SoluteSpecies(
    name="Mg",
    D0=1.49e-5,       # m^2/s
    Qdiff=120.5e3,    # J/mol
)

Si = SoluteSpecies(
    name="Si",
    D0=1.38e-5,       # m^2/s
    Qdiff=117.6e3,    # J/mol
)


# ============================================================================
# Precipitate phases
# ============================================================================

beta = PrecipitatePhase(
    name="beta",
    stoichiometry={
        "Mg": 2,
        "Si": 1,
    },
    atomic_volume=0.0215,     # nm^3/atom
    gamma=0.15e-18,            # J/nm^2
    gb_barrier_factor=0.5,
    thermodynamics=SolubilityProductThermodynamics(
        aa=7097.639,
        bb=2.342283,
        cc=0.0,
        use_xy_factor=True,
    ),
)


betapp = PrecipitatePhase(
    name="beta_pp",
    stoichiometry={
        "Mg": 5,
        "Si": 6,
    },
    atomic_volume=0.0192,     # nm^3/atom

    # Provisional beta'' interfacial energy.
    gamma=0.15e-18,            # J/nm^2

    gb_barrier_factor=1.0,
    thermodynamics=SolubilityProductThermodynamics(
        aa=16670.694,
        bb=4.651,
        cc=0.0,
        use_xy_factor=True,
    ),
)


# ============================================================================
# Alloy system
# ============================================================================

alloy = AlloySystem(
    solvent="Al",
    matrix_atomic_volume=1.66e-2,      # nm^3/atom
    matrix_lattice_parameter=0.405,    # nm
    solutes=(Mg, Si),
    phases=(beta, betapp),
)


# ============================================================================
# Initial matrix composition
# ============================================================================

initial_composition = {
    "Mg": 0.78798e-2,
    "Si": 0.87404e-2,
}
