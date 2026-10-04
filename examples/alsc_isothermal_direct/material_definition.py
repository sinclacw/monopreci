from monopole_precip.phases import (
    AlloySystem,
    PrecipitatePhase,
    SolubilityProductThermodynamics,
    SoluteSpecies,
)


# ============================================================================
# Solute species
# ============================================================================

Sc = SoluteSpecies(
    name="Sc",
    D0=5.3e-4,       # m^2/s
    Qdiff=174.0e3,   # J/mol
)


# ============================================================================
# Precipitate phase
# ============================================================================

Al3Sc = PrecipitatePhase(
    name="Al3Sc",
    stoichiometry={
        "Al": 3,
        "Sc": 1,
    },
    atomic_volume=0.0173,   # nm^3/atom
    gamma=0.12e-18,         # J/nm^2 = 0.12 J/m^2
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
    matrix_atomic_volume=0.0166,      # nm^3/atom
    matrix_lattice_parameter=0.405,   # nm
    solutes=(Sc,),
    phases=(Al3Sc,),
)


# ============================================================================
# Initial matrix composition
# ============================================================================

# Al-0.50 at.% Sc.
initial_composition = {
    "Sc": 0.005,
}
