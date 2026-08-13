from pathlib import Path

import numpy as np

from wheelbot.control import UnifiedController
from wheelbot.simulation import Simulation, initial_position
from wheelbot.experiments import Experiment
from wheelbot.linear_model import (
    roll_reaction_wheel_model, pitch_wheeled_pendulum_model,
)


def _passive(name, roll=0.0, pitch=0.0):
    controller = UnifiedController()

    def control(state):
        action, phase = controller.control(state, mode="passive")
        return action, 0.0, phase

    return Experiment(name, 2.0, controller,
                      lambda m: initial_position(m, roll=roll, pitch=pitch),
                      control)


def _measure_lambda(rows, angle_key, rate_key, lo=0.09, hi=0.5):
    t = np.array([r["time"] for r in rows])
    angle = np.array([r[angle_key] for r in rows])
    rate = np.array([r[rate_key] for r in rows])
    accel = np.gradient(rate, t)
    mask = (np.abs(angle) > lo) & (np.abs(angle) < hi)
    lam_sq = np.median(accel[mask] / angle[mask])
    return float(np.sqrt(max(lam_sq, 0.0))), int(mask.sum())


def main() -> None:
    out = Path("results")
    out.mkdir(exist_ok=True)
    rows_pitch = Simulation().run(_passive("val_pitch", pitch=5.0),
                                  render=False, output=out / "val_pitch.csv")
    rows_roll = Simulation().run(_passive("val_roll", roll=5.0),
                                 render=False, output=out / "val_roll.csv")

    lam_pitch, n_p = _measure_lambda(rows_pitch, "pitch", "pitch_rate")
    lam_roll, n_r = _measure_lambda(rows_roll, "roll", "roll_rate")
    pred_roll = roll_reaction_wheel_model().unstable_poles.real.max()
    pred_pitch = pitch_wheeled_pendulum_model().unstable_poles.real.max()

    print("=== Open-loop divergence: model vs MuJoCo passive fall ===")
    print(f"{'axis':6s} {'model lambda':>13s} {'MuJoCo lambda':>15s} {'error':>8s}   fit pts")
    for axis, pred, meas, n in (("roll", pred_roll, lam_roll, n_r),
                                ("pitch", pred_pitch, lam_pitch, n_p)):
        err = 100.0 * (meas - pred) / pred
        print(f"{axis:6s} {pred:13.2f} {meas:15.2f} {err:7.1f}%   {n}")


if __name__ == "__main__":
    main()
