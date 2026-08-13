import csv
import math
from pathlib import Path

import mujoco

from wheelbot.control.balance import BalanceController
from wheelbot.simulation import Simulation, State, initial_position


LEANS = (0.04, 0.06, 0.08, 0.10)
SPEEDS = (0.30, 0.50)
SETTLE_TIME = 1.5
TRIAL_TIME = 3.0


def _trial(roll_target, command_speed):

    simulation = Simulation()
    controller = BalanceController()
    controller.reset()
    simulation.reset(initial_position(simulation.model))

    leans, yaw_rates, speeds, wheel_speeds = [], [], [], []
    while simulation.data.time < TRIAL_TIME:
        state = State(simulation)
        role_action = controller.control(
            state,
            x_target=state.x,
            vx_target=command_speed,
            roll_target=roll_target,
            heading_target=state.yaw,
            path_direction=None,
            speed_limit=command_speed,
        )
        simulation.apply(simulation.roles.to_physical(role_action))
        mujoco.mj_step(simulation.model, simulation.data)
        if state.time >= SETTLE_TIME:
            leans.append(state.balancing_tilt)
            yaw_rates.append(state.yaw_rate)
            speeds.append(math.hypot(state.vx, state.vy))
            wheel_speeds.append(abs(state.balancing_wheel_speed))

    n = len(leans)
    mean_lean = sum(leans) / n
    mean_rate = sum(yaw_rates) / n
    mean_speed = sum(speeds) / n
    denominator = mean_speed * mean_lean
    return {
        "commanded_lean_rad": roll_target,
        "commanded_speed_m_s": command_speed,
        "measured_lean_rad": mean_lean,
        "measured_speed_m_s": mean_speed,
        "measured_yaw_rate_rad_s": mean_rate,
        "coefficient_per_m": mean_rate / denominator if abs(denominator) > 1e-9 else float("nan"),
        "peak_reaction_wheel_rad_s": max(wheel_speeds),
        "samples": n,
    }


def sweep():
    return [
        _trial(lean, speed)
        for speed in SPEEDS
        for lean in LEANS
    ]


def main(output=Path("results") / "yaw_identification.csv"):
    trials = sweep()

    print("=== Lean-to-yaw-rate identification ===")
    print("  psi_dot = c * v * phi ;  c reported in m^-1 (curvature per unit lean)")
    print(f"  {'cmd v':>6} {'cmd lean':>9} {'meas v':>8} {'meas lean':>10} "
          f"{'yaw rate':>9} {'c':>8} {'peak wheel':>11}")
    for t in trials:
        print(f"  {t['commanded_speed_m_s']:6.2f} {t['commanded_lean_rad']:9.3f} "
              f"{t['measured_speed_m_s']:8.3f} {t['measured_lean_rad']:10.4f} "
              f"{t['measured_yaw_rate_rad_s']:9.4f} {t['coefficient_per_m']:8.2f} "
              f"{t['peak_reaction_wheel_rad_s']:11.0f}")

    values = [t["coefficient_per_m"] for t in trials if math.isfinite(t["coefficient_per_m"])]
    if values:
        mean = sum(values) / len(values)
        spread = max(values) - min(values)
        print(f"\n  c = {mean:.1f} m^-1, spread {spread:.1f} over "
              f"{len(values)} trials ({100 * spread / mean:.0f}% of the mean)")
        print("  A coefficient constant across BOTH sweeps confirms the bilinear")
        print("  rolling-contact form. A coefficient falling with speed would")
        print("  instead indicate a coordinated banked turn (psi_dot = g tan(phi)/v),")
        print("  which would remove the reaction wheel's need to hold the lean.")
        by_speed = {}
        for t in trials:
            by_speed.setdefault(t["commanded_speed_m_s"], []).append(t["coefficient_per_m"])
        for speed, group in sorted(by_speed.items()):
            print(f"    at {speed:.2f} m/s: c = {sum(group) / len(group):.1f} m^-1")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=trials[0])
        writer.writeheader()
        writer.writerows(trials)
    print(f"  saved {output}")


if __name__ == "__main__":
    main()
