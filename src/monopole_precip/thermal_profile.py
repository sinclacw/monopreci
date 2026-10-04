import numpy as np


def _build_isothermal_hold(
    temperature_C,
    isothermal_time_s,
    isothermal_segment_time_s,
):
    """Build a piecewise-constant isothermal hold schedule."""

    isothermal_time_s = float(isothermal_time_s)
    isothermal_segment_time_s = float(isothermal_segment_time_s)

    if (
        not np.isfinite(isothermal_time_s)
        or isothermal_time_s <= 0.0
    ):
        raise ValueError(
            "isothermal_time_s must be positive and finite."
        )

    if (
        not np.isfinite(isothermal_segment_time_s)
        or isothermal_segment_time_s <= 0.0
    ):
        raise ValueError(
            "isothermal_segment_time_s must be positive and finite."
        )

    # Number of complete equal-duration hold segments.
    n_full = int(
        np.floor(
            isothermal_time_s
            / isothermal_segment_time_s
        )
    )

    remainder = (
        isothermal_time_s
        - n_full * isothermal_segment_time_s
    )

    segment_durations = np.full(
        n_full,
        isothermal_segment_time_s,
        dtype=float,
    )

    # Preserve the requested total hold time exactly when it is not an
    # integer multiple of isothermal_segment_time_s.
    if remainder > 1.0e-14 * max(1.0, isothermal_time_s):
        segment_durations = np.concatenate(
            [
                segment_durations,
                np.array([remainder], dtype=float),
            ]
        )

    # This is mainly defensive; for positive hold time the code above will
    # normally already have produced at least one segment.
    if len(segment_durations) == 0:
        segment_durations = np.array(
            [isothermal_time_s],
            dtype=float,
        )

    temperatures = np.full(
        len(segment_durations),
        float(temperature_C) + 273.15,
        dtype=float,
    )

    return temperatures, segment_durations


def linear_cooling(
    T_start_C,
    T_end_C,
    cooling_rate_C_per_s,
    temperature_step_C=1.0,
    isothermal_time_s=None,
    isothermal_segment_time_s=1.0,
):
    """
    Generate a piecewise-constant thermal schedule.

    Cooling only:
        T_start_C > T_end_C and isothermal_time_s is None

        A piecewise-constant approximation to linear cooling is generated
        using midpoint temperatures.

    Cooling followed by an isothermal hold:
        T_start_C > T_end_C and isothermal_time_s is not None

        The material is first cooled from T_start_C to T_end_C, then held
        exactly at T_end_C for isothermal_time_s.

    Isothermal only:
        T_start_C == T_end_C

        The temperature is held constant for isothermal_time_s.

    Parameters
    ----------
    T_start_C : float
        Starting temperature in degrees Celsius.

    T_end_C : float
        Final cooling / hold temperature in degrees Celsius.

    cooling_rate_C_per_s : float
        Cooling rate in degrees Celsius per second.
        Used only when T_start_C > T_end_C.

    temperature_step_C : float, optional
        Temperature interval per thermal segment during cooling.

    isothermal_time_s : float or None, optional
        If T_start_C == T_end_C, this is the duration of the isothermal
        calculation and is required.

        If T_start_C > T_end_C, a non-None value appends an isothermal hold
        at exactly T_end_C after cooling.

    isothermal_segment_time_s : float, optional
        Duration of each thermal-history segment during the isothermal hold.
        This controls thermal-history segmentation spacing; it does not set the
        adaptive internal integration timestep.

    Returns
    -------
    temperatures : ndarray
        Segment temperatures in kelvin.

    segment_durations : ndarray
        Duration of each segment in seconds.
    """

    T_start_C = float(T_start_C)
    T_end_C = float(T_end_C)

    # ---------------------------------------------------------
    # Isothermal-only case
    # ---------------------------------------------------------
    if np.isclose(
        T_start_C,
        T_end_C,
        rtol=0.0,
        atol=1.0e-12,
    ):
        if isothermal_time_s is None:
            raise ValueError(
                "isothermal_time_s must be specified when "
                "T_start_C == T_end_C."
            )

        return _build_isothermal_hold(
            temperature_C=T_end_C,
            isothermal_time_s=isothermal_time_s,
            isothermal_segment_time_s=isothermal_segment_time_s,
        )

    # ---------------------------------------------------------
    # Cooling case
    # ---------------------------------------------------------
    if T_start_C < T_end_C:
        raise ValueError(
            "T_start_C must be greater than or equal to T_end_C."
        )

    cooling_rate_C_per_s = float(cooling_rate_C_per_s)
    temperature_step_C = float(temperature_step_C)

    if (
        not np.isfinite(cooling_rate_C_per_s)
        or cooling_rate_C_per_s <= 0.0
    ):
        raise ValueError(
            "cooling_rate_C_per_s must be positive and finite."
        )

    if (
        not np.isfinite(temperature_step_C)
        or temperature_step_C <= 0.0
    ):
        raise ValueError(
            "temperature_step_C must be positive and finite."
        )

    # Midpoint-temperature approximation to linear cooling.
    # Example for 550 -> 200 C with dT=1 C:
    #   first segment T = 549.5 C
    #   last  segment T = 200.5 C
    # each lasting 1 s for a cooling rate of 1 C/s.
    temperatures_C = np.arange(
        T_start_C - 0.5 * temperature_step_C,
        T_end_C,
        -temperature_step_C,
    )

    temperatures = temperatures_C + 273.15

    segment_durations = np.full(
        len(temperatures),
        temperature_step_C / cooling_rate_C_per_s,
        dtype=float,
    )

    # ---------------------------------------------------------
    # Optional isothermal hold at the final cooling temperature
    # ---------------------------------------------------------
    if isothermal_time_s is not None:
        hold_temperatures, hold_durations = _build_isothermal_hold(
            temperature_C=T_end_C,
            isothermal_time_s=isothermal_time_s,
            isothermal_segment_time_s=isothermal_segment_time_s,
        )

        temperatures = np.concatenate(
            [temperatures, hold_temperatures]
        )

        segment_durations = np.concatenate(
            [segment_durations, hold_durations]
        )

    return temperatures, segment_durations
