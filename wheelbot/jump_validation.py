import csv
import math
from pathlib import Path

import mujoco

from wheelbot.control.balance import BalanceController
from wheelbot.control import UnifiedController
from wheelbot.experiments import Experiment
from wheelbot.jump_planner import DEFAULT_CONFIG, plan_jump
from wheelbot.physical_design import DESIGN
from wheelbot.simulation import (
    Simulation,
    State,
    contact_height,
    initial_position,
)


VALIDATION_FEATURES = (
    ("low", 0.024, 0.026),
    ("medium", 0.036, 0.035),
    ("high", 0.044, 0.042),
)
IDENTIFICATION_PRELOADS = tuple(
    millimetres / 1000.0 for millimetres in range(12, 21)
)
APPROACH_VALIDATION_SPEEDS = (0.50, 0.60, 0.70, 0.80)


def identify_launch_model():

    trials = [_release_trial(preload) for preload in IDENTIFICATION_PRELOADS]
    lower_envelope = min(
        trial["observed_retained_energy"] for trial in trials
    )
    retained_energy_lower_bound = (
        math.floor(1000.0 * lower_envelope) / 1000.0
    )
    return {
        "trials": trials,
        "retained_energy_lower_bound": retained_energy_lower_bound,
        "release_delay_upper_bound": max(
            trial["release_delay_s"] for trial in trials
        ),
    }


def _release_trial(preload):

    simulation = Simulation()
    qpos = initial_position(simulation.model)
    qpos[simulation.jump_qpos] = preload
    qpos[2] = 0.0
    qpos[2] = contact_height(simulation.model, qpos)
    simulation.reset(qpos)

    initial_height = State(simulation).z
    peak_height = initial_height
    release_delay = None
    while simulation.data.time < 0.5:
        state = State(simulation)
        if not state.grounded and release_delay is None:
            release_delay = state.time
        if release_delay is not None:
            peak_height = max(peak_height, state.z)
            if (
                state.grounded
                and state.time - release_delay >= 0.020
            ):
                break
        simulation.apply((0.0, 0.0, 0.0))
        mujoco.mj_step(simulation.model, simulation.data)
    if release_delay is None:
        raise RuntimeError(f"no lift-off observed for preload {preload}")

    retained_energy = (
        2.0
        * DESIGN.total_mass
        * DESIGN.gravity
        * (peak_height - initial_height)
        / (DESIGN.spring_stiffness * preload**2)
    )
    return {
        "preload_m": preload,
        "initial_height_m": initial_height,
        "peak_height_m": peak_height,
        "release_delay_s": release_delay,
        "observed_retained_energy": retained_energy,
    }


def validate_approach_tracking():

    return [
        _approach_trial(speed) for speed in APPROACH_VALIDATION_SPEEDS
    ]


def _approach_trial(command_speed):
    simulation = Simulation()
    controller = BalanceController()
    controller.reset()
    simulation.reset(initial_position(simulation.model))
    speeds = []
    tilts = []
    supported = 0
    samples = 0

    while simulation.data.time < 4.0:
        state = State(simulation)
        role_action = controller.control(
            state,
            x_target=state.x,
            vx_target=command_speed,
            heading_target=0.0,
            path_direction=(1.0, 0.0),
            speed_limit=command_speed,
        )
        physical_action = simulation.roles.to_physical(role_action)
        simulation.apply(physical_action)
        mujoco.mj_step(simulation.model, simulation.data)
        if state.time >= 2.0:
            speeds.append(state.vx)
            tilts.append(
                max(
                    abs(state.driving_tilt),
                    abs(state.balancing_tilt),
                )
            )
            supported += int(state.supported_on_wheel)
            samples += 1

    rmse = math.sqrt(
        sum((speed - command_speed) ** 2 for speed in speeds) / samples
    )
    result = {
        "command_speed_m_s": command_speed,
        "mean_speed_m_s": sum(speeds) / samples,
        "speed_rmse_m_s": rmse,
        "maximum_tilt_rad": max(tilts),
        "supported_fraction": supported / samples,
    }
    result["passed"] = int(
        result["speed_rmse_m_s"] <= 0.010
        and result["maximum_tilt_rad"] <= 0.030
        and result["supported_fraction"] >= 0.95
    )
    return result


def validate_feature(name, width, height):

    center = 0.68
    destination = 1.05
    feature = center, width, height
    plan = plan_jump(feature)
    controller = UnifiedController()
    simulation = Simulation()
    collision_samples = 0

    def control(state):
        nonlocal collision_samples
        collision_samples += int(_has_obstacle_contact(simulation))
        action, phase = controller.control(
            state,
            mode="jump",
            feature=feature,
            destination=destination,
            heading_target=0.0,
        )
        return action, destination, phase

    experiment = Experiment(
        f"jump_validation_{name}",
        8.0,
        controller,
        lambda model: initial_position(model),
        control,
        obstacle=(feature,),
    )
    rows = simulation.run(experiment)
    phases = {row["phase"] for row in rows}

    half_width = width / 2.0
    radius = DESIGN.wheel_radius
    measured_clearances = []
    for row in rows:
        relative = row["x"] - center
        if -half_width - radius <= relative <= half_width + radius:
            gap = max(
                -half_width - relative,
                relative - half_width,
                0.0,
            )
            wheel_drop = math.sqrt(max(0.0, radius**2 - gap**2))
            measured_clearances.append(
                row["z"]
                - DESIGN.wheel_offset
                - wheel_drop
                - height
            )

    result = {
        "feature": name,
        "width_m": width,
        "height_m": height,
        "planned_speed_m_s": plan["horizontal"],
        "planned_vertical_speed_m_s": plan["vertical"],
        "planned_preload_m": plan["preload"],
        "planned_peak_m": plan["peak"],
        "simulated_peak_m": max(row["z"] for row in rows),
        "planned_clearance_m": plan["clearance"],
        "sampled_clearance_m": min(measured_clearances),
        "observed_spring_efficiency": (
            2.0
            * DESIGN.total_mass
            * DESIGN.gravity
            * (
                max(row["z"] for row in rows)
                - DESIGN.upright_height
            )
            / (
                DESIGN.spring_stiffness
                * plan["preload"] ** 2
            )
        ),
        "obstacle_contacts": collision_samples,
        "completed": int("complete" in phases),
        "flight_observed": int("flight" in phases),
    }
    result["passed"] = int(
        result["completed"]
        and result["flight_observed"]
        and result["obstacle_contacts"] == 0
        and result["sampled_clearance_m"]
        >= DEFAULT_CONFIG.requirements.clearance_margin - 0.001
    )
    return result


def _has_obstacle_contact(simulation):

    obstacle_geoms = {
        simulation.geom("demo_obstacle"),
        simulation.geom("demo_feature_1"),
        simulation.geom("demo_feature_2"),
        simulation.geom("demo_feature_3"),
    }
    for index in range(simulation.data.ncon):
        contact = simulation.data.contact[index]
        pair = {contact.geom1, contact.geom2}
        if not pair & obstacle_geoms:
            continue
        other = pair - obstacle_geoms
        if other and simulation.floor_geom not in other:
            return True
    return False


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def run_validation(output_directory=Path("results")):
    identification = identify_launch_model()
    approach = validate_approach_tracking()
    results = [
        validate_feature(name, width, height)
        for name, width, height in VALIDATION_FEATURES
    ]
    _write_rows(
        output_directory / "jump_launch_identification.csv",
        identification["trials"],
    )
    _write_rows(
        output_directory / "jump_approach_validation.csv",
        approach,
    )
    _write_rows(
        output_directory / "jump_planner_validation.csv",
        results,
    )
    return identification, approach, results


def main():
    identification, approach, results = run_validation()
    print(
        "identified launch model: "
        f"eta>={identification['retained_energy_lower_bound']:.3f}, "
        "release delay<="
        f"{1000 * identification['release_delay_upper_bound']:.0f} ms"
    )
    for result in approach:
        print(
            f"approach {result['command_speed_m_s']:.2f} m/s: "
            f"RMSE={result['speed_rmse_m_s']:.4f} m/s, "
            f"passed={bool(result['passed'])}"
        )
    for result in results:
        print(
            f"{result['feature']}: "
            f"clearance={1000 * result['sampled_clearance_m']:.2f} mm, "
            f"contacts={result['obstacle_contacts']}, "
            f"passed={bool(result['passed'])}"
        )
    frozen = DEFAULT_CONFIG.launch
    identification_matches = (
        identification["retained_energy_lower_bound"]
        == frozen.retained_energy_lower_bound
        and math.isclose(
            identification["release_delay_upper_bound"],
            frozen.release_delay_upper_bound,
            abs_tol=1e-12,
        )
    )
    if (
        not identification_matches
        or not all(result["passed"] for result in approach)
        or not all(result["passed"] for result in results)
    ):
        raise SystemExit("jump-planner validation failed")
    print("saved jump identification and validation evidence in results/")


if __name__ == "__main__":
    main()
