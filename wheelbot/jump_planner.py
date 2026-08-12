from dataclasses import dataclass, field
from functools import lru_cache
import math

from wheelbot.physical_design import DESIGN, PhysicalDesign


_SOLVER_TOLERANCE = 1e-10
_SOLVER_ITERATIONS = 128


@dataclass(frozen=True)
class IdentifiedLaunchModel:

    retained_energy_lower_bound: float = 0.425
    minimum_identified_preload: float = 0.012
    maximum_identified_preload: float = 0.020
    release_delay_upper_bound: float = 0.016
    maximum_validated_approach_speed: float = 0.80

    def __post_init__(self):
        positive = (
            self.retained_energy_lower_bound,
            self.minimum_identified_preload,
            self.maximum_identified_preload,
            self.release_delay_upper_bound,
            self.maximum_validated_approach_speed,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("identified launch-model values must be positive")
        if self.retained_energy_lower_bound > 1.0:
            raise ValueError("retained energy cannot exceed stored energy")
        if (
            self.minimum_identified_preload
            > self.maximum_identified_preload
        ):
            raise ValueError("invalid identified preload interval")


@dataclass(frozen=True)
class JumpRequirements:

    clearance_margin: float = 0.010

    def __post_init__(self):
        if (
            not math.isfinite(self.clearance_margin)
            or self.clearance_margin < 0.0
        ):
            raise ValueError("clearance margin must be finite and non-negative")


@dataclass(frozen=True)
class JumpPlannerConfig:

    design: PhysicalDesign = field(default_factory=lambda: DESIGN)
    launch: IdentifiedLaunchModel = field(
        default_factory=IdentifiedLaunchModel
    )
    requirements: JumpRequirements = field(default_factory=JumpRequirements)

    def __post_init__(self):
        if (
            self.launch.maximum_identified_preload
            > self.design.maximum_static_preload + _SOLVER_TOLERANCE
        ):
            raise ValueError(
                "identified preload exceeds the physical actuator limit"
            )


DEFAULT_CONFIG = JumpPlannerConfig()


def plan_jump(feature, config=DEFAULT_CONFIG):

    values = tuple(map(float, feature))
    if len(values) != 3:
        raise ValueError("a terrain feature must be (centre, width, height)")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("terrain feature values must be finite")
    center, width, height = values
    if width <= 0.0 or height < 0.0:
        raise ValueError("feature width must be positive and height non-negative")
    return dict(_plan_jump(values, config))


@lru_cache(maxsize=256)
def _plan_jump(feature, config):
    center, width, height = feature
    design = config.design
    launch = config.launch
    maximum_design_speed = launch.maximum_validated_approach_speed
    maximum_preload = min(
        launch.maximum_identified_preload,
        design.maximum_static_preload,
    )
    maximum_vertical = _vertical_speed_for_preload(maximum_preload, config)
    minimum_vertical = _vertical_speed_for_preload(
        launch.minimum_identified_preload,
        config,
    )

    def state(speed_squared):
        speed = math.sqrt(speed_squared)
        required_peak = _required_peak_height(
            width,
            height,
            speed,
            config,
        )
        vertical = math.sqrt(
            max(
                0.0,
                2.0
                * design.gravity
                * (required_peak - design.upright_height),
            )
        )
        launched_vertical = max(vertical, minimum_vertical)
        energy = 0.5 * design.total_mass * (
            speed_squared + launched_vertical * launched_vertical
        )
        feasible = vertical <= maximum_vertical
        return energy, vertical, required_peak, feasible

    lower = _SOLVER_TOLERANCE
    upper = maximum_design_speed**2
    if not state(upper)[3]:
        raise ValueError(
            "no feasible jump inside the identified spring and approach-speed "
            "envelope"
        )

    if not state(lower)[3]:
        infeasible = lower
        feasible = upper
        for _ in range(_SOLVER_ITERATIONS):
            midpoint = 0.5 * (infeasible + feasible)
            if state(midpoint)[3]:
                feasible = midpoint
            else:
                infeasible = midpoint
            if feasible - infeasible <= _SOLVER_TOLERANCE:
                break
        lower = feasible

    optimum, iterations = _golden_section_minimize(
        lambda speed_squared: state(speed_squared)[0],
        lower,
        upper,
        tolerance=_SOLVER_TOLERANCE,
        maximum_iterations=_SOLVER_ITERATIONS,
    )
    choices = (lower, optimum, upper)
    speed_squared = min(choices, key=lambda value: state(value)[0])
    energy, required_vertical, required_peak, _ = state(speed_squared)
    design_speed = math.sqrt(speed_squared)

    preload = max(
        launch.minimum_identified_preload,
        _preload_for_vertical_speed(required_vertical, config),
    )
    if preload > maximum_preload + _SOLVER_TOLERANCE:
        raise ValueError("jump requires more than the available spring stroke")
    preload = min(preload, maximum_preload)

    vertical = _vertical_speed_for_preload(preload, config)
    peak = design.upright_height + vertical**2 / (2.0 * design.gravity)
    crossing_time = vertical / design.gravity
    flight_time = 2.0 * crossing_time
    takeoff = center - design_speed * crossing_time
    landing = center + design_speed * crossing_time

    uninflated_peak = _required_peak_height(
        width,
        height,
        design_speed,
        config,
        clearance_margin=0.0,
    )
    clearance = peak - uninflated_peak
    energy = 0.5 * design.total_mass * (
        design_speed**2 + vertical**2
    )

    launch_gate_margin = 0.003
    low_speed = _SOLVER_TOLERANCE
    high_speed = design_speed
    for _ in range(_SOLVER_ITERATIONS):
        mid_speed = 0.5 * (low_speed + high_speed)
        gate_clearance = peak - _required_peak_height(
            width, height, mid_speed, config, clearance_margin=0.0
        )
        if gate_clearance >= launch_gate_margin:
            high_speed = mid_speed
        else:
            low_speed = mid_speed
        if high_speed - low_speed <= _SOLVER_TOLERANCE:
            break
    minimum_horizontal = high_speed

    return {
        "takeoff": takeoff,
        "landing": landing,
        "clearance_entry": center - 0.5 * width - design.wheel_radius,
        "clearance_exit": center + 0.5 * width + design.wheel_radius,
        "horizontal": design_speed,
        "minimum_horizontal": minimum_horizontal,
        "maximum_horizontal": maximum_design_speed,
        "vertical": vertical,
        "flight_time": flight_time,
        "crossing_time": crossing_time,
        "peak": peak,
        "clearance": clearance,
        "energy": energy,
        "cost": energy,
        "preload": preload,
        "release_delay": launch.release_delay_upper_bound,
        "method": "convex_minimum_energy",
        "converged": True,
        "iterations": iterations,
    }


def _required_peak_height(
    width,
    height,
    horizontal_speed,
    config,
    *,
    clearance_margin=None,
):

    margin = (
        config.requirements.clearance_margin
        if clearance_margin is None
        else float(clearance_margin)
    )
    half_width = 0.5 * width
    design = config.design
    radius = design.wheel_radius
    gravity_term = design.gravity / (2.0 * horizontal_speed**2)

    def flank_value(offset):
        wheel_drop = math.sqrt(max(0.0, radius**2 - offset**2))
        position = half_width + offset
        return wheel_drop + gravity_term * position**2

    def flank_slope(offset):
        wheel_slope = -offset / math.sqrt(
            max(1e-30, radius**2 - offset**2)
        )
        ballistic_slope = (
            design.gravity
            * (half_width + offset)
            / horizontal_speed**2
        )
        return wheel_slope + ballistic_slope

    low = 0.0
    high = radius
    for _ in range(_SOLVER_ITERATIONS):
        midpoint = 0.5 * (low + high)
        if flank_slope(midpoint) > 0.0:
            low = midpoint
        else:
            high = midpoint
        if high - low <= _SOLVER_TOLERANCE:
            break
    stationary = 0.5 * (low + high)

    envelope_height = max(
        radius + gravity_term * half_width**2,
        flank_value(stationary),
        flank_value(radius),
    )
    return (
        height
        + margin
        + design.wheel_offset
        + envelope_height
    )


def _golden_section_minimize(
    objective,
    lower,
    upper,
    *,
    tolerance,
    maximum_iterations,
):

    inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
    left = float(lower)
    right = float(upper)
    c = right - inverse_phi * (right - left)
    d = left + inverse_phi * (right - left)
    fc = objective(c)
    fd = objective(d)
    iterations = 0
    while (
        right - left > tolerance
        and iterations < maximum_iterations
    ):
        if fc <= fd:
            right = d
            d = c
            fd = fc
            c = right - inverse_phi * (right - left)
            fc = objective(c)
        else:
            left = c
            c = d
            fc = fd
            d = left + inverse_phi * (right - left)
            fd = objective(d)
        iterations += 1
    return 0.5 * (left + right), iterations


def _preload_for_vertical_speed(vertical_speed, config):
    design = config.design
    return math.sqrt(
        design.total_mass * vertical_speed**2
        / (
            config.launch.retained_energy_lower_bound
            * design.spring_stiffness
        )
    )


def _vertical_speed_for_preload(preload, config):
    design = config.design
    return math.sqrt(
        config.launch.retained_energy_lower_bound
        * design.spring_stiffness
        * preload**2
        / design.total_mass
    )
