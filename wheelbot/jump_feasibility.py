from pathlib import Path

import numpy as np

from wheelbot.physical_design import DESIGN
from wheelbot.jump_planner import (
    plan_jump, DEFAULT_CONFIG,
    _required_peak_height, _vertical_speed_for_preload,
)


def _achieved_apex(preload, config=DEFAULT_CONFIG):
    v = _vertical_speed_for_preload(preload, config)
    return config.design.upright_height + v * v / (2.0 * config.design.gravity)


def feasibility_grid(feature, config=DEFAULT_CONFIG, n=161):
    _, width, height = map(float, feature)
    v_max = config.launch.maximum_validated_approach_speed
    x_max = min(config.launch.maximum_identified_preload,
                config.design.maximum_static_preload)
    speeds = np.linspace(0.15, v_max, n)
    preloads = np.linspace(0.0, x_max, n)
    required = np.array([
        _required_peak_height(width, height, s, config, clearance_margin=0.0)
        for s in speeds
    ])
    apex = np.array([_achieved_apex(p, config) for p in preloads])
    clearance = 1000.0 * (apex[:, None] - required[None, :])
    return speeds, preloads, clearance, x_max


def robustness(feature, config=DEFAULT_CONFIG):
    plan = plan_jump(feature, config)
    _, width, height = map(float, feature)
    ds, dp = plan["horizontal"], plan["preload"]
    x_max = min(config.launch.maximum_identified_preload,
                config.design.maximum_static_preload)

    apex_dp = _achieved_apex(dp, config)
    speeds = np.linspace(0.15, config.launch.maximum_validated_approach_speed, 4000)
    req = np.array([_required_peak_height(width, height, s, config, clearance_margin=0.0)
                    for s in speeds])
    clearing = speeds[apex_dp >= req]
    min_speed = float(clearing.min()) if clearing.size else float("nan")

    req_ds = _required_peak_height(width, height, ds, config, clearance_margin=0.0)
    preloads = np.linspace(0.0, x_max, 4000)
    apex = np.array([_achieved_apex(p, config) for p in preloads])
    clearing_p = preloads[apex >= req_ds]
    min_preload = float(clearing_p.min()) if clearing_p.size else float("nan")

    return {
        "design_speed": ds, "design_preload": dp,
        "min_clearing_speed": min_speed, "speed_slack": ds - min_speed,
        "min_clearing_preload": min_preload, "preload_slack": dp - min_preload,
        "max_preload": x_max,
    }


def measure_flight_attitude(feature, destination=1.40):
    from wheelbot.control import UnifiedController
    from wheelbot.simulation import Simulation, initial_position
    from wheelbot.experiments import Experiment

    controller = UnifiedController()
    commands = (("jump", tuple(map(float, feature)), destination, 0.0),)

    def control(state):
        action, phase = controller.control(state, mode="mission", commands=commands)
        return action, destination, phase

    exp = Experiment("flight_attitude", 24.0, controller,
                     lambda m: initial_position(m), control, obstacle=(feature,))
    rows = Simulation().run(exp, render=False, output=None)
    airborne = [r for r in rows if r.get("phase") in ("release", "flight")
                or r.get("grounded", 1) == 0]
    deg = np.degrees
    result = {
        "airborne_samples": len(airborne),
        "flight_pitch_max_deg": deg(max((abs(r["pitch"]) for r in airborne), default=0.0)),
        "flight_roll_max_deg": deg(max((abs(r["roll"]) for r in airborne), default=0.0)),
        "flight_yaw_max_deg": deg(max((abs(r["yaw"]) for r in airborne), default=0.0)),
        "final_pitch_deg": deg(rows[-1]["pitch"]),
        "final_roll_deg": deg(rows[-1]["roll"]),
        "final_phase": rows[-1]["phase"],
    }
    return result


def landing_state(feature, config=DEFAULT_CONFIG):
    plan = plan_jump(feature, config)
    landing_x = plan["landing"]
    clearance_exit = plan["clearance_exit"]
    landing_margin = landing_x - clearance_exit
    v_vertical = plan["vertical"]
    v_horizontal = plan["horizontal"]
    impact_speed = float(np.hypot(v_vertical, v_horizontal))
    impact_angle = float(np.degrees(np.arctan2(v_vertical, v_horizontal)))
    return {
        "landing_x": landing_x,
        "clearance_exit": clearance_exit,
        "landing_margin": landing_margin,
        "lands_clear": landing_margin > 0.0,
        "impact_vertical": v_vertical,
        "impact_horizontal": v_horizontal,
        "impact_speed": impact_speed,
        "impact_angle_deg": impact_angle,
    }


def plot_feature(feature, config=DEFAULT_CONFIG, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speeds, preloads, clearance, x_max = feasibility_grid(feature, config)
    plan = plan_jump(feature, config)
    S, P = np.meshgrid(speeds, preloads * 1000.0)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    pcm = ax.pcolormesh(S, P, clearance, shading="auto", cmap="RdYlGn",
                        vmin=-30, vmax=30)
    fig.colorbar(pcm, ax=ax, label="clearance = apex - required (mm)")
    ax.contour(S, P, clearance, levels=[0.0], colors="k", linewidths=1.8)
    ax.contour(S, P, clearance, levels=[10.0], colors="k", linewidths=1.0,
               linestyles="--")
    ax.plot(plan["horizontal"], plan["preload"] * 1000.0, "b*", markersize=15,
            label="planner design point")
    ax.set_xlabel("launch horizontal speed (m/s)")
    ax.set_ylabel("spring preload (mm)")
    c, w, h = feature
    ax.set_title(f"Jump feasibility envelope  (obstacle w={w*1000:.0f} h={h*1000:.0f} mm)")
    ax.legend(loc="lower left")
    ax.text(0.98, 0.03, "solid = clears,  dashed = +10 mm margin",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    fig.tight_layout()
    out = Path(path) if path else Path("results") / f"jump_feasibility_{int(h*1000)}mm.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main():
    for feature in [(1.18, 0.024, 0.026), (1.08, 0.036, 0.035)]:
        r = robustness(feature)
        s = landing_state(feature)
        print(f"=== feasibility for obstacle {feature} ===")
        print(f"  design point : speed {r['design_speed']:.3f} m/s, "
              f"preload {r['design_preload']*1000:.1f} mm  (max {r['max_preload']*1000:.1f} mm)")
        print(f"  PHYSICAL slack (still clears):")
        print(f"    speed can drop to {r['min_clearing_speed']:.3f} m/s "
              f"-> slack {r['speed_slack']*1000:.0f} mm/s below design")
        print(f"    preload can drop to {r['min_clearing_preload']*1000:.1f} mm "
              f"-> slack {r['preload_slack']*1000:.1f} mm below design")
        print(f"  landing: lands {s['landing_margin']*1000:+.1f} mm past far edge "
              f"({'clears' if s['lands_clear'] else 'LANDS ON OBSTACLE'}); "
              f"impact {s['impact_speed']:.2f} m/s at {s['impact_angle_deg']:.0f} deg")
        out = plot_feature(feature)
        print(f"  figure -> {out}")
    fa = measure_flight_attitude((0.70, 0.024, 0.026))
    print("\n=== flight attitude (ballistic-assumption check) ===")
    print(f"  max in-flight |pitch|={fa['flight_pitch_max_deg']:.1f} deg "
          f"|roll|={fa['flight_roll_max_deg']:.1f} deg |yaw|={fa['flight_yaw_max_deg']:.1f} deg; "
          f"lands at pitch={fa['final_pitch_deg']:.1f} deg ({fa['final_phase']})")


if __name__ == "__main__":
    main()
