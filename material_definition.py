"""
Reference material definition for Mono'Preci.

This file defines an Al-Sc alloy containing an Al3Sc precipitate phase.
It is intended to accompany the reference ``run_configuration.py`` in the
repository root.

Material properties are expressed using the internal units used by
Mono'Preci:

Length             : nm
Energy              : J
Diffusion prefactor : m^2/s
Activation energy   : J/mol
Atomic volume       : nm^3/atom
Interfacial energy  : J/nm^2

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

Sc = SoluteSpecies(
    name="Sc",

    # D = D0 * exp(-Qdiff / RT)
    D0=5.3e-4,       # m^2/s
    Qdiff=174.0e3,   # J/mol
)


# ============================================================================
# Precipitate phases
# ============================================================================

Al3Sc = PrecipitatePhase(
    name="Al3Sc",

    # Al3Sc stoichiometry.
    stoichiometry={
        "Al": 3,
        "Sc": 1,
    },

    # Atomic volume of the precipitate [nm^3/atom].
    atomic_volume=0.0173,

    # Matrix/precipitate interfacial energy [J/nm^2].
    #
    # 0.12 J/m^2 = 0.12e-18 J/nm^2.
    gamma=0.12e-18,

    # None indicates that no phase-specific heterogeneous-nucleation
    # barrier factor is defined for this reference material.
    gb_barrier_factor=None,

    thermodynamics=SolubilityProductThermodynamics(
        aa=2576.0,
        bb=0.1158,
        cc=-43018.0,
        use_xy_factor=True,
        solvent_stoichiometric_weight=3.0,
    ),
)


# ============================================================================
# Alloy system
# ============================================================================

alloy = AlloySystem(
    solvent="Al",

    # Matrix atomic volume [nm^3/atom].
    matrix_atomic_volume=0.0166,

    # Matrix lattice parameter [nm].
    matrix_lattice_parameter=0.405,

    solutes=(Sc,),
    phases=(Al3Sc,),
)


# ============================================================================
# Initial matrix composition
# ============================================================================

# Al-0.50 at.% Sc.
#
# Compositions are atomic/site fractions.
initial_composition = {
    "Sc": 0.005,
}
