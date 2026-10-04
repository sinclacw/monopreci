"""
Reference run configuration for Mono'Preci.

This file defines the spatial, numerical, nucleation, transport, and output
controls used by the ``monopreci`` command-line driver.

Material properties and initial alloy composition are defined separately in
``material_definition.py``. The thermal history, random seed, and output
filenames are specified on the command line.

The values below provide a compact direct-monopole reference calculation.
Individual examples under ``examples/`` contain configurations appropriate
to the calculations demonstrated there.

Units
-----
Length      : nm
Time        : s
Temperature : supplied through the command line in degrees Celsius
"""

import numpy as np


# ============================================================================
# Simulation cell and initial particle population
# ============================================================================

# Periodic simulation-cell dimensions [nm].
box = np.array([80.0, 80.0, 80.0], dtype=float)

# Initial precipitate centres [nm].
#
# Shape: (N, 3)
# An empty array corresponds to an initially precipitate-free matrix.
positions = np.empty((0, 3), dtype=float)

# Initial precipitate radii [nm].
#
# Shape: (N,)
radii0 = np.empty(0, dtype=float)

# Precipitate-phase index for each initial particle.
#
# Phase indices correspond to the ordering of alloy.phases in
# material_definition.py.
phase_id0 = np.empty(0, dtype=np.int32)


# ============================================================================
# Adaptive time integration
# ============================================================================

# Initial trial timestep [s].
dt_initial = 100.0

# Maximum timestep permitted for the first accepted internal step [s].
#
# This is useful when nucleation begins from an initially particle-free
# supersaturated matrix.
first_step_dt_max = 0.01

# Global maximum and minimum internal timesteps [s].
dt_max = 10.0
dt_min = 1.0e-8

# Fractional-growth safety factor used by the adaptive timestep controller.
dt_safety = 0.1

# Duration of each segment used to represent an isothermal hold [s].
#
# This controls thermal-history segmentation, not the internal adaptive
# integration timestep.
isothermal_segment_time_s = 50.0


# ============================================================================
# Nucleation
# ============================================================================

# Enable stochastic nucleation.
enable_nucleation = True

# Additional geometric separation [nm] required when placing a new nucleus.
# This affects spatial placement only; it does not modify the CNT barrier
# or nucleation rate.
safety_buffer = 0.0

# Apply the incubation-time correction to the nucleation rate.
apply_incubation_correction = True

# Upper target for the expected number of nucleation events within an
# integration step. The timestep is reduced when necessary to respect this
# target.
max_expected_nucleation_events = 200.0

# Account for spatial variations in the local nucleation rate.
nucleation_spatial_rate_correction = True

# Target relative standard error used when estimating the spatially averaged
# nucleation rate.
nucleation_spatial_rate_rse_target = 0.20

# Effective-sample-size target used when sampling spatial nucleation sites.
nucleation_placement_ess_per_event = 1.5

# Write detailed nucleation diagnostics to a separate CSV file.
save_nucleation_diagnostics = False


# ============================================================================
# Optional planar grain-boundary region
# ============================================================================

# Point lying on the grain-boundary plane [nm].
gb_point = 0.5 * box

# Unit normal to the grain-boundary plane.
gb_normal = np.array([1.0, 0.0, 0.0], dtype=float)

# Width of the grain-boundary nucleation region [nm].
#
# Set to zero when grain-boundary nucleation is not required.
gb_thickness = 0.0


# ============================================================================
# Direct periodic Green-function formulation
# ============================================================================

# Optional screening length [nm] for the direct periodic Green function.
#
# None selects the unscreened periodic Laplace Green function.
screening_length = None


# ============================================================================
# Particle removal
# ============================================================================

# Particles below this radius are removed [nm].
R_remove = 0.25

# Additional radius interval used by the early-removal treatment [nm].
R_remove_buffer = 0.10


# ============================================================================
# Scale-separated transport
# ============================================================================

# False:
#     Direct quasistatic periodic monopole formulation.
#
# True:
#     Scale-separated formulation with quasistatic short-range monopole
#     fields and a transient long-range concentration field.
hybrid_transport = False

# Matching length ell [nm].
#
# This defines the separation between the quasistatic short-range field and
# the transient long-range field.
hybrid_matching_length_nm = 40.0

# Short-range interaction cutoff expressed as a multiple of ell.
hybrid_short_cutoff_factor = 3.0

# Target spacing of the periodic long-range FFT grid [nm].
#
# The actual spacing may differ slightly because FFT-friendly grid dimensions
# are selected automatically.
hybrid_grid_spacing_nm = 5.0


# ============================================================================
# Initial long-range concentration field
# ============================================================================

# Initial long-range field:
#
#     "uniform" : spatially uniform initial matrix composition
#     "cosine"  : add a zero-mean cosine perturbation
#
# These settings are relevant only when hybrid_transport = True.
initial_lr_field = "uniform"

# Cartesian axis for an optional cosine perturbation:
#     0 -> x
#     1 -> y
#     2 -> z
initial_lr_cosine_axis = 0

# Integer Fourier mode number of the cosine perturbation.
initial_lr_cosine_mode = 1

# Cosine amplitude for each solute.
#
# The amplitudes are dimensionless fractional perturbations relative to the
# mean concentration. For species s,
#
#     C_s(r, 0) = cbar_s * [1 + A_s * phi(r)]
#
# where phi is the selected zero-mean cosine mode and A_s is the value
# specified here.
#
# Example: a 10% modulation of the Sc concentration:
# initial_lr_cosine_amplitude = {"Sc": 0.10}
initial_lr_cosine_amplitude = {}


# ============================================================================
# Long-range-field output
# ============================================================================

# Save native long-range concentration-grid snapshots at particle-dump times.
save_hybrid_long_range_field_snapshots = False

# Store a sampled long-range-field history during the calculation.
store_hybrid_long_range_history = False

# Segment interval for the sampled long-range-field history.
hybrid_long_range_history_every_segments = 1

# Spatial sampling interval for the stored long-range-field history [nm].
hybrid_long_range_history_spacing_nm = 4.0

# Numeric precision used for the stored long-range-field history.
hybrid_long_range_history_dtype = "float32"


# ============================================================================
# General output and reporting
# ============================================================================

# Save the restart/current-state NPZ file every this many thermal segments.
save_every_segments = 10

# Optional requested particle-dump times [s].
#
# None uses segment-based dumping controlled by dump_seg_every.
dump_times_s = None

# Always include the final state in the particle dump.
dump_include_final = True

# Particle-dump interval in thermal segments when dump_times_s is None.
dump_seg_every = 1

# Internal-step interval for detailed progress reporting.
print_every = 20

# Thermal-segment interval for segment-level progress reporting.
print_every_segments = 1

# Enable detailed solver profiling output.
profile_solver = False

# Internal-step interval for the detailed solver-history save mechanism.
#
# A large value effectively disables frequent internal-step history output
# while retaining the supported configuration interface.
save_every_internal_steps = 100000
