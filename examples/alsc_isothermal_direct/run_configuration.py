import numpy as np

# ============================================================================
# Geometry and initial particles
# ============================================================================

box = np.array([80.0, 80.0, 80.0])  # nm

# Begin with an initially precipitate-free matrix.
positions = np.empty((0, 3), dtype=float)
radii0 = np.empty(0, dtype=float)
phase_id0 = np.empty(0, dtype=np.int32)


# ============================================================================
# Time integration
# ============================================================================

dt_initial = 100.0
first_step_dt_max = 0.01

dt_max = 10.0
dt_min = 1.0e-8
dt_safety = 0.1
isothermal_segment_time_s = 50.0


# ============================================================================
# Nucleation
# ============================================================================

enable_nucleation = True
safety_buffer = 0.0
apply_incubation_correction = True
max_expected_nucleation_events = 200.0

nucleation_spatial_rate_correction = True
nucleation_spatial_rate_rse_target = 0.20
nucleation_placement_ess_per_event = 1.5
save_nucleation_diagnostics = True

# Grain-boundary nucleation is inactive in this example.
# These parameters are retained because they are part of the run configuration.
gb_point = 0.5 * box
gb_normal = np.array([1.0, 0.0, 0.0], dtype=float)
gb_thickness = 0.0


# ============================================================================
# Particle removal
# ============================================================================

screening_length = None
R_remove = 0.25
R_remove_buffer = 0.10


# ============================================================================
# Transport formulation
# ============================================================================

# Use the direct quasistatic monopole formulation.
hybrid_transport = False

# The scale-separated transport parameters below are inactive for this example
# but are retained because they are part of the run configuration.
hybrid_matching_length_nm = 10.0
hybrid_short_cutoff_factor = 3.0
hybrid_grid_spacing_nm = 3.0

initial_lr_field = "uniform"
initial_lr_cosine_axis = 0
initial_lr_cosine_mode = 1
initial_lr_cosine_amplitude = {}

save_hybrid_long_range_field_snapshots = False

store_hybrid_long_range_history = False
hybrid_long_range_history_every_segments = 1
hybrid_long_range_history_spacing_nm = 4.0
hybrid_long_range_history_dtype = "float32"


# ============================================================================
# Output and diagnostics
# ============================================================================

save_every_segments = 10

dump_times_s = None
dump_include_final = True
dump_seg_every = 1

print_every = 20
print_every_segments = 1
profile_solver = False

save_every_internal_steps = 100000
