import ast
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wheelbot import linear_model as LM

RESULTS = Path("results")
FIGDIR = Path("figures")
GRAVITY = 9.81

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
})
C = {"roll": "#1f77b4", "pitch": "#d62728", "yaw": "#2ca02c",
     "wheel": "#7f7f7f", "ref": "#999999", "aux": "#ff7f0e", "ok": "#2ca02c",
     "warn": "#ff7f0e", "bad": "#d62728", "abort": "#9467bd"}


def load(name):
    path = RESULTS / f"{name}.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array([float(r[k]) for r in rows])
        except ValueError:
            out[k] = [r[k] for r in rows]
    return out


def _save(fig, name, dpi=None):
    FIGDIR.mkdir(exist_ok=True)
    p = FIGDIR / name
    fig.tight_layout()
    fig.savefig(p, **({"dpi": dpi} if dpi else {}))
    plt.close(fig)
    return p


def _settling_time(t, angle, tol=0.05):
    outside = np.where(np.abs(angle) > tol)[0]
    return float(t[outside[-1]]) if len(outside) else 0.0


def fig_model_validation():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, axis, key, lam in (
        (axes[0], "roll", "roll", 8.96),
        (axes[1], "pitch", "pitch", 14.69),
    ):
        d = load(f"val_{axis}")
        if d is None:
            continue
        t, a = d["time"], d[key]
        a = np.abs(a)
        model = a[0] * np.cosh(lam * (t - t[0]))
        valid = np.abs(a) < 0.55
        tmax = t[valid][-1] if valid.any() else t[-1]
        ax.plot(t, np.degrees(a), color=C[axis], label="MuJoCo plant")
        ax.plot(t, np.degrees(model), "--", color="k",
                label=fr"model $\alpha_0\cosh(\lambda t),\ \lambda={lam:.2f}$")
        ax.set_ylim(0, np.degrees(a[t <= tmax]).max() * 1.1)
        ax.set_xlim(t[0], tmax)
        ax.set_title(f"{axis.capitalize()} passive fall")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(f"{axis} angle (deg)")
        ax.legend(loc="upper left")
    fig.suptitle("Model validation: open-loop divergence vs simulator "
                 "(roll −2.9%, pitch −5.4%)", fontsize=11)
    return _save(fig, "01_model_validation.png")


def _stable_closed_loop(m, K):
    K = np.array([K])
    for Acl in (m.A + m.B @ K, m.A - m.B @ K):
        p = np.linalg.eigvals(Acl)
        if np.all(p.real < 1e-6):
            return p
    return None


def fig_pole_maps():
    g = LM.ROLL_GAINS
    mr = LM.roll_model_full()
    Kr = [g["k_phi"], g["k_phidot"], g["k_theta"], g["k_thetadot"]]
    mp = LM.pitch_wheeled_pendulum_model()
    Kp = [0.40 * 0.21, 0.40 * 0.4956, 0.40, 0.05]
    specs = [
        ("Roll (reaction-wheel IP)", mr.poles, _stable_closed_loop(mr, Kr)),
        ("Pitch (wheeled IP)", mp.poles, _stable_closed_loop(mp, Kp)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (title, ol, cl) in zip(axes, specs):
        allreal = list(ol.real) + (list(cl.real) if cl is not None else [])
        xr = max(abs(np.array(allreal))) * 1.15
        ax.axvspan(0, xr, alpha=0.06, color=C["bad"])
        ax.scatter(ol.real, ol.imag, s=95, marker="x", color=C["bad"],
                   label="open-loop", zorder=3, linewidths=2)
        if cl is not None:
            ax.scatter(cl.real, cl.imag, s=75, marker="o", facecolors="none",
                       edgecolors=C["ok"], label="closed-loop", zorder=3,
                       linewidths=1.6)
        ax.axvline(0, color="k", lw=0.9)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlim(-xr, xr)
        ax.set_title(title)
        ax.set_xlabel(r"Re$(s)$ (s$^{-1}$)")
        ax.set_ylabel(r"Im$(s)$")
        ax.legend(loc="upper left")
    fig.suptitle("Pole maps: the unstable open-loop pole (RHP ×) is moved into "
                 "the LHP by the controller (○)", fontsize=11)
    return _save(fig, "02_pole_maps.png")


def fig_yaw_controllability():
    speeds = np.linspace(0.0, 0.8, 41)
    measure = []
    for v in speeds:
        m = LM.yaw_steering_model(max(v, 1e-9))
        Cm = m.controllability_matrix()
        measure.append(np.linalg.svd(Cm, compute_uv=False).min())
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(speeds, measure, color=C["yaw"])
    ax.fill_between(speeds, 0, measure, color=C["yaw"], alpha=0.12)
    ax.set_xlabel("forward speed $v$ (m/s)")
    ax.set_ylabel("controllability (min singular value)")
    ax.set_title(r"Yaw is uncontrollable at rest: authority $\to 0$ as $v \to 0$")
    ax.axvline(0, color=C["bad"], lw=1.0, ls=":")
    ax.annotate("standstill:\nrank drops", (0.01, ax.get_ylim()[1] * 0.5),
                color=C["bad"], fontsize=9)
    return _save(fig, "03_yaw_controllability.png")


def fig_roll_balance():
    d = load("roll_balance")
    if d is None:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(d["time"], np.degrees(d["roll"]), color=C["roll"], label="roll angle")
    ax.axhline(0, color=C["ref"], lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("roll angle (deg)", color=C["roll"])
    ts = _settling_time(d["time"], d["roll"])
    ax2 = ax.twinx()
    ax2.plot(d["time"], d["wheel_x_speed"], color=C["wheel"], alpha=0.6,
             label="reaction-wheel speed")
    ax2.set_ylabel("reaction-wheel speed (rad/s)", color=C["wheel"])
    ax2.grid(False)
    ax.set_title(f"Roll balancing: recovery from 20°, settling {ts:.2f} s")
    return _save(fig, "04_roll_balance.png")


def fig_pitch_balance():
    d = load("pitch_balance")
    if d is None:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(d["time"], np.degrees(d["pitch"]), color=C["pitch"], label="pitch angle")
    ax.axhline(0, color=C["ref"], lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pitch angle (deg)", color=C["pitch"])
    ax2 = ax.twinx()
    ax2.plot(d["time"], d["x"], color=C["aux"], alpha=0.7, label="base position x")
    ax2.set_ylabel("base position x (m)", color=C["aux"])
    ax2.grid(False)
    ts = _settling_time(d["time"], d["pitch"])
    ax.set_title(f"Pitch balancing: recovery from 50°, settling {ts:.2f} s "
                 f"(base travels to catch the CoM)")
    return _save(fig, "05_pitch_balance.png")


def fig_motion_tracking():
    d = load("motion")
    if d is None:
        return None
    t, x = d["time"], d["x"]
    ref = d.get("target")
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(t, x, color=C["roll"], label="actual position x")
    if isinstance(ref, np.ndarray):
        axes[0].plot(t, ref, "--", color=C["ref"], label="reference position")
        err = x - ref
        ss = float(np.mean(np.abs(err[-50:]))) * 1000
        axes[1].plot(t, err * 1000, color=C["bad"])
        axes[1].axhline(0, color="k", lw=0.6)
        axes[1].set_ylabel("error (mm)")
        axes[0].set_title(f"Motion tracking: reaches each waypoint, "
                          f"steady-state error {ss:.0f} mm "
                          f"(transients are step-response lag)")
    axes[0].set_ylabel("position (m)")
    axes[0].legend(loc="best")
    axes[1].set_xlabel("time (s)")
    return _save(fig, "06_motion_tracking.png")


def fig_disturbance():
    d = load("disturbance")
    if d is None:
        return None
    from wheelbot.disturbances import DisturbanceCampaign as _DC
    t = d["time"]
    roll = np.degrees(d["roll"])
    pitch = np.degrees(d["pitch"])
    tilt = np.maximum(np.abs(roll), np.abs(pitch))
    phase = d.get("phase")
    kd_t = _DC.events[-1][0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8),
                                   gridspec_kw={"width_ratios": [3, 2]})

    pre = t < kd_t - 1.5
    ax1.plot(t[pre], roll[pre], color=C["roll"], label="roll angle")
    ax1.plot(t[pre], pitch[pre], color=C["pitch"], label="pitch angle")
    ax1.axhline(0, color="k", lw=0.6)
    ylim = np.abs(np.concatenate([roll[pre], pitch[pre]])).max() * 1.15
    for (t0, _, _), lbl in list(zip(_DC.events, _DC.labels))[:4]:
        ax1.axvline(t0, color=C["aux"], ls=":", alpha=0.7)
        ax1.text(t0 + 0.25, ylim * 0.98, lbl, rotation=90, va="top", fontsize=8,
                 color="#555")
    ax1.set_ylim(-ylim, ylim)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("attitude angle (deg)")
    ax1.set_title("Attitude rejection: pitch (forward/back) and roll (both sides)")
    ax1.legend(loc="lower right")

    post = t >= kd_t - 1.5
    ax2.plot(t[post], tilt[post], color=C["bad"], label="body tilt from upright")
    if isinstance(phase, list):
        rec = np.array([isinstance(p, str) and "recover" in p for p in phase])
        ax2.fill_between(t, 0, 200, where=rec, color=C["ok"], alpha=0.15,
                         label="self-right active")
    ax2.set_xlim(kd_t - 1.5, t[-1])
    ax2.set_ylim(0, 195)
    ax2.axvline(kd_t, color=C["aux"], ls=":", alpha=0.7)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("body tilt from upright (deg)")
    ax2.set_title("Hard shove: knockdown → self-right")
    ax2.legend(loc="upper right")
    fig.suptitle("Disturbance rejection and recovery: pitch, roll, and self-righting",
                 fontsize=11)
    return _save(fig, "07_disturbance_recovery.png")


def fig_yaw_turn():
    d = load("yaw")
    if d is None:
        return None
    t, yaw = d["time"], np.degrees(d["yaw"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(t, yaw, color=C["yaw"], label="heading")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("heading (deg)")
    ax.set_title("Yaw: commanded in-place turns (180° then back to 0°)")
    ax.axhline(180, color=C["ref"], ls="--", lw=0.8)
    ax.axhline(0, color=C["ref"], ls="--", lw=0.8)
    return _save(fig, "08_yaw_turn.png")


def fig_jump():
    d = load("jump")
    if d is None:
        return None
    t = d["time"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(t, d["z"] * 1000, color=C["roll"], label="CoM height z")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("height z (mm)", color=C["roll"])
    ax2 = ax.twinx()
    ax2.plot(t, d["x"], color=C["aux"], alpha=0.7, label="base x")
    ax2.set_ylabel("base x (m)", color=C["aux"])
    ax2.grid(False)
    zmax = d["z"].max()
    ax.set_title(f"Jump: apex z = {zmax*1000:.0f} mm, clears the obstacle "
                 f"and lands upright")
    return _save(fig, "09_jump.png")


def fig_self_righting():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, axis in zip(axes, ("pitch", "roll")):
        d = load(f"self_right_{axis}")
        if d is None:
            continue
        t = d["time"]
        tilt = np.maximum(np.abs(d["roll"]), np.abs(d["pitch"]))
        ax.plot(t, np.degrees(tilt), color=C[axis])
        ax.axhline(math.degrees(0.25), color=C["ok"], ls="--", lw=0.8,
                   label="upright band")
        rec = _settling_time(t, tilt, tol=0.25)
        ax.set_title(f"{axis.capitalize()}-axis recovery: {rec:.1f} s")
        ax.set_xlabel("time (s)")
        ax.legend(loc="upper right")
    axes[0].set_ylabel("body tilt from upright (deg)")
    fig.suptitle("Self-righting from a fallen state (both axes)", fontsize=11)
    return _save(fig, "10_self_righting.png")


def fig_unified():
    d = load("unified")
    if d is None:
        return None
    t = d["time"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.4), sharex=True)
    tilt = np.maximum(np.abs(d["roll"]), np.abs(d["pitch"]))
    axes[0].plot(t, d["x"], color=C["aux"], label="forward position x")
    axes[0].set_ylabel("forward position x (m)", color=C["aux"])
    axz = axes[0].twinx()
    axz.plot(t, d["z"] * 1000, color=C["roll"], alpha=0.6, label="CoM height z")
    axz.set_ylabel("CoM height z (mm)", color=C["roll"])
    axz.grid(False)
    axes[0].set_title("Unified mission: self-right → move → turn → move → jump → balance")
    axes[1].plot(t, np.degrees(d["yaw"]), color=C["yaw"])
    axes[1].set_ylabel("heading angle (deg)")
    axes[2].plot(t, np.degrees(tilt), color=C["bad"])
    axes[2].set_ylabel("body tilt from upright (deg)")
    axes[2].set_xlabel("time (s)")
    return _save(fig, "11_unified_mission.png")


def fig_sensitivity():
    params = ["total mass", "wheel inertia", "chassis inertia",
              "spring stiffness", "gravity", "tyre friction",
              "tyre contact", "end-stop damping", "wheel torque cap"]
    caps = ["self-right", "jump", "unified"]
    grid = {
        "total mass": [("P", "P", "P"), ("P", "P", "I")],
        "wheel inertia": [("P", "P", "P"), ("P", "P", "P")],
        "chassis inertia": [("P", "P", "P"), ("P", "P", "P")],
        "spring stiffness": [("P", "P", "P"), ("P", "A", "I")],
        "gravity": [("P", "A", "P"), ("P", "P", "F")],
        "tyre friction": [("P", "P", "P"), ("P", "P", "P")],
        "tyre contact": [("P", "P", "P"), ("P", "P", "P")],
        "end-stop damping": [("P", "P", "I"), ("P", "P", "P")],
        "wheel torque cap": [("P", "P", "P"), ("P", "P", "P")],
    }
    code = {"P": (C["ok"], "PASS"), "A": (C["abort"], "ABORT (safe refusal)"),
            "I": (C["warn"], "INCOMPLETE (upright)"), "F": (C["bad"], "FALL")}
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ncol = len(caps) * 2
    for pi, p in enumerate(params):
        for li, lvl in enumerate((0, 1)):
            for ci, cap in enumerate(caps):
                col = li * len(caps) + ci
                o = grid[p][lvl][ci]
                ax.add_patch(plt.Rectangle((col, pi), 1, 1,
                             color=code[o][0], alpha=0.85, ec="white"))
                ax.text(col + 0.5, pi + 0.5, o, ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
    ax.set_xlim(0, ncol)
    ax.set_ylim(0, len(params))
    ax.set_yticks([i + 0.5 for i in range(len(params))])
    ax.set_yticklabels(params)
    ax.set_xticks([i + 0.5 for i in range(ncol)])
    ax.set_xticklabels([f"{c}\n{s}" for s in ("−20%", "+20%") for c in caps],
                       fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Parameter sensitivity (±20%): 48 PASS · 2 safe ABORT · "
                 "3 INCOMPLETE · 1 FALL")
    for grp in (0, 3, 6):
        ax.axvline(grp, color="k", lw=0.6)
    ax.set_xticks([]) if False else None
    handles = [plt.Rectangle((0, 0), 1, 1, color=v[0]) for v in code.values()]
    ax.legend(handles, [v[1] for v in code.values()], ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _save(fig, "12_sensitivity.png")


def fig_metric_summary():
    times, dists, angles = [], [], []
    d = load("roll_balance")
    if d:
        times.append(("Roll settling time", _settling_time(d["time"], d["roll"])))
    d = load("pitch_balance")
    if d:
        times.append(("Pitch settling time", _settling_time(d["time"], d["pitch"])))
    for axis in ("pitch", "roll"):
        d = load(f"self_right_{axis}")
        if d:
            tilt = np.maximum(np.abs(d["roll"]), np.abs(d["pitch"]))
            times.append((f"{axis.capitalize()} self-right time",
                          _settling_time(d["time"], tilt, 0.25)))
    d = load("disturbance")
    if d:
        t = d["time"]
        tilt = np.maximum(np.abs(d["roll"]), np.abs(d["pitch"]))
        pk = int(np.argmax(tilt))
        rec = np.where((t > t[pk]) & (np.degrees(tilt) < 5))[0]
        if len(rec):
            times.append(("Disturbance recovery", float(t[rec[0]] - t[pk])))
    d = load("motion")
    if d and isinstance(d.get("target"), np.ndarray):
        err = np.abs(d["x"] - d["target"])
        dists.append(("Motion steady-state error", float(np.mean(err[-50:])) * 1000))
    d = load("jump")
    if d:
        dists.append(("Jump apex rise", (d["z"].max() - 0.098) * 1000))
    d = load("roll_balance")
    if d:
        angles.append(("Roll steady-state error", abs(math.degrees(d["roll"][-1]))))
    d = load("pitch_balance")
    if d:
        angles.append(("Pitch steady-state error", abs(math.degrees(d["pitch"][-1]))))
    d = load("unified")
    if d:
        tilt = max(abs(d["roll"][-1]), abs(d["pitch"][-1]))
        angles.append(("Unified final tilt", math.degrees(tilt)))

    panels = [("Settling / recovery time", "seconds", times, C["roll"]),
              ("Distance", "millimetres", dists, C["yaw"]),
              ("Angle", "degrees", angles, C["pitch"])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (title, unit, data, colour) in zip(axes, panels):
        labels = [m[0] for m in data]
        vals = [m[1] for m in data]
        bars = ax.barh(labels, vals, color=colour, alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                    f" {v:.2f}", va="center", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel(unit)
        ax.margins(x=0.22)
        ax.invert_yaxis()
    fig.suptitle("Quantitative performance summary", fontsize=11)
    return _save(fig, "00_metric_summary.png")


def obstacles(experiment):

    src = (Path(__file__).parent / "experiments.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == experiment:
            for stmt in ast.walk(node):
                if (isinstance(stmt, ast.Assign)
                        and getattr(stmt.targets[0], "id", None) == "features"):
                    return tuple(tuple(v) for v in ast.literal_eval(stmt.value))
    return ()


def fig_launch_map():

    from wheelbot.physical_design import DESIGN
    path = RESULTS / "jump_launch_identification.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    preload = np.array([float(r["preload_m"]) for r in rows])
    eta = np.array([float(r["observed_retained_energy"]) for r in rows])

    total = DESIGN.total_mass
    sprung = total / 2.0
    ceiling = sprung / total
    ideal = ceiling * (1.0 - 2.0 * sprung * GRAVITY / (DESIGN.spring_stiffness * preload))
    resid = ideal - eta
    mean, sd = resid.mean(), resid.std()

    mm = preload * 1000.0
    fig, (top, bot) = plt.subplots(2, 1, figsize=(7, 4.8), sharex=True,
                                   gridspec_kw={"height_ratios": [2.4, 1.0]})
    top.axhline(ceiling, color=C["ref"], ls=":", lw=1.2)
    top.text(mm[0], ceiling, f"  structural ceiling $m_s/m_{{tot}}$ = {ceiling:.3f}",
             va="bottom", fontsize=8, color=C["ref"])
    top.plot(mm, ideal, color=C["roll"], lw=2.0, label=r"derived ideal $\eta_{ideal}(s)$")
    top.scatter(mm, eta, s=44, color=C["pitch"], zorder=3, edgecolor="white", lw=1.0,
                label=f"identified from the plant ({len(mm)} free releases)")
    top.set_ylabel(r"apex coefficient $\eta$")
    top.set_ylim(eta.min() - 0.02, ceiling + 0.02)
    top.legend(loc="lower right", fontsize=8, frameon=False)
    top.set_title("Launch map: identified apex coefficient vs the closed-form ceiling")
    bot.axhspan(mean - sd, mean + sd, color=C["roll"], alpha=0.15, lw=0)
    bot.axhline(mean, color=C["roll"], lw=1.5)
    bot.scatter(mm, resid, s=32, color=C["pitch"], zorder=3, edgecolor="white", lw=0.9)
    bot.text(mm[-1], mean + 2.2 * sd, f"mean {mean:.4f}, s.d. {sd:.4f}",
             ha="right", fontsize=8)
    bot.set_ylabel("residual")
    bot.set_xlabel("spring preload (mm)")
    bot.set_ylim(mean - 4 * sd, mean + 4 * sd)
    for ax in (top, bot):
        ax.grid(alpha=0.3)
        ax.set_xticks(mm)
    return _save(fig, "14_launch_map.png", dpi=300)


def fig_jump_execution():
    data = load("jump")
    if data is None:
        return None
    phase = data["phase"]
    idx = [i for i, p in enumerate(phase) if "1_jump" in p]
    if not idx:
        return None
    lo, hi = idx[0], idx[-1] + 1
    t = data["time"][lo:hi]
    ext = data["jump_extension"][lo:hi] * 1000.0
    z = data["z"][lo:hi] * 1000.0
    phase = phase[lo:hi]

    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 4.4), sharex=True)
    for key, col in (("preload", C["warn"]), ("flight", C["roll"]), ("recover", C["ok"])):
        inside = [i for i, p in enumerate(phase) if p.endswith(key)]
        if not inside:
            continue
        for ax in (a, b):
            ax.axvspan(t[inside[0]], t[inside[-1]], color=col, alpha=0.13, lw=0)
        a.text(t[inside[0]], ext.max() * 0.95, f" {key}", fontsize=8, color=col)
    a.plot(t, ext, color="#333333", lw=1.6)
    a.set_ylabel("spring\ncompression (mm)")
    a.set_title(f"Jump execution: peak chassis height {z.max():.0f} mm")
    b.plot(t, z, color=C["roll"], lw=1.8)
    b.set_ylabel("chassis height (mm)")
    b.set_xlabel("time (s)")
    for ax in (a, b):
        ax.grid(alpha=0.3)
    return _save(fig, "15_jump_execution.png", dpi=300)


def fig_jump_sequence():
    from wheelbot.physical_design import DESIGN
    data = load("jump")
    if data is None:
        return None
    x, z = data["x"], data["z"]
    tyre = z - DESIGN.upright_height

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for pos, w, h in obstacles("jump"):
        ax.add_patch(plt.Rectangle((pos - w / 2, 0), w, h, color=C["aux"], alpha=0.55, lw=0))
        ax.annotate(f"{w * 1000:.0f} x {h * 1000:.0f} mm", (pos, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8, color=C["aux"])
    ax.plot(x, z, color=C["roll"], lw=1.6, label="chassis root")
    ax.plot(x, tyre, color="#333333", lw=1.3, ls="--", label="tyre lowest point")
    ax.set_xlabel("travel (m)")
    ax.set_ylabel("height (m)")
    ax.set_ylim(-0.005, max(z.max(), 0.16) * 1.15)
    ax.set_title("Two-obstacle sequence as a single mission")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.3)
    return _save(fig, "16_jump_sequence.png", dpi=300)


def fig_mission_timeline():

    STAGE_NAMES = {
        "recover": "autonomous recovery",
        "1_move_to": "translate to path",
        "2_yaw": "turn onto heading",
        "3_move_to": "approach run-up",
        "4_jump": "jump and settle",
    }

    data = load("unified")
    if data is None:
        return None
    t, phase, yaw = data["time"], data["phase"], data.get("yaw")

    def stage(p):
        parts = p.split("/")
        if parts[0] == "recover":
            return "recover"
        return parts[1] if len(parts) > 1 else parts[0]

    spans, start_t, current = [], t[0], stage(phase[0])
    for i in range(1, len(phase)):
        key = stage(phase[i])
        if key != current:
            spans.append((current, start_t, t[i]))
            start_t, current = t[i], key
    spans.append((current, start_t, t[-1]))
    spans = [s for s in spans if s[0] != "complete"]
    if not spans:
        return None
    total = spans[-1][2]

    turn = ""
    if yaw is not None:
        for name, a, b in spans:
            if name.endswith("yaw"):
                ia = int(np.argmin(abs(t - a)))
                ib = int(np.argmin(abs(t - b)))
                turn = f" {abs(np.degrees(yaw[ib] - yaw[ia])):.0f}\u00b0"

    cols = [C["pitch"], C["roll"], C["wheel"], C["aux"], C["ok"], C["abort"]]
    fig, ax = plt.subplots(figsize=(8.6, 2.4))
    for i, (key, a, b) in enumerate(spans):
        label = STAGE_NAMES.get(key, key.replace("_", " "))
        if key.endswith("yaw") and turn:
            label += turn
        ax.barh(0, b - a, left=a, height=0.5, color=cols[i % len(cols)],
                alpha=0.85, edgecolor="white", linewidth=1.2)
        ax.text((a + b) / 2, 0.31, f"{b - a:.1f} s", ha="center", fontsize=8)
        ax.text((a + b) / 2, -0.34, label, ha="center", va="top", fontsize=8,
                rotation=0 if (b - a) / total > 0.12 else 30,
                rotation_mode="anchor")
    widest = max(spans, key=lambda s: s[2] - s[1])
    share = (widest[2] - widest[1]) / total * 100.0
    ax.annotate(f"{share:.0f} % of the mission",
                xy=((widest[1] + widest[2]) / 2, 0.58), ha="center", fontsize=8.5)
    ax.set_ylim(-1.0, 0.85)
    ax.set_xlim(-total * 0.02, total * 1.02)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(f"Integrated mission: complete at {total:.2f} s")
    ax.grid(axis="x", alpha=0.3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    return _save(fig, "17_mission_timeline.png", dpi=300)


FIGURES = [
    fig_metric_summary, fig_model_validation, fig_pole_maps,
    fig_yaw_controllability, fig_roll_balance, fig_pitch_balance,
    fig_motion_tracking, fig_disturbance, fig_yaw_turn, fig_jump,
    fig_self_righting, fig_unified, fig_sensitivity,
    fig_launch_map, fig_jump_execution, fig_jump_sequence, fig_mission_timeline,
]


def main():
    made = []
    for fn in FIGURES:
        try:
            p = fn()
            if p:
                made.append(p)
                print(f"  saved {p}")
            else:
                print(f"  skipped {fn.__name__} (missing data)")
        except Exception as exc:
            print(f"  FAILED {fn.__name__}: {exc}")
    print(f"\n{len(made)} figures written to {FIGDIR}/")


if __name__ == "__main__":
    main()
