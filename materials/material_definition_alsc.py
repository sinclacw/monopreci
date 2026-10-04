"""
Al-Sc material definition for Mono'Preci.

This file defines the Al-Sc material system used by the Al-Sc examples
distributed with Mono'Preci.

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

Sc = SoluteSpecies(
    name="Sc",

    # D = D0 * exp(-Qdiff / RT)
    D0=5.3e-4,       # m^2/s
    Qdiff=174.0e3,   # J/mol
)


# ============================================================================
# Precipitate phase
# ============================================================================

Al3Sc = PrecipitatePhase(
    name="Al3Sc",

    # Al3Sc stoichiometry.
    stoichiometry={
        "Al": 3,
        "Sc": 1,
    },

    # Atomic volume [nm^3/atom].
    atomic_volume=0.0173,

    # Matrix/precipitate interfacial energy [J/nm^2].
    #
    # 0.12 J/m^2 = 0.12e-18 J/nm^2.
    gamma=0.12e-18,

    # No phase-specific grain-boundary nucleation factor is defined here.
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
# This is the composition used by the direct isothermal Al-Sc example.
# Individual calculations may use a different initial composition by
# providing an appropriate material_definition.py in the calculation
# directory.
initial_composition = {
    "Sc": 0.005,
}
