import mujoco

from wheelbot import experiments
from wheelbot.simulation import Simulation

ROBOT_BODIES = ("wheel_x_end", "wheel_x", "wheel_y_end", "wheel_y")
WHEEL_BODIES = ("wheel_x", "wheel_y")
END_BODIES = ("wheel_x_end", "wheel_y_end")


def _bid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _jid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def perturb_mass(model, f):
    for name in ROBOT_BODIES:
        i = _bid(model, name)
        model.body_mass[i] *= f
        model.body_inertia[i] *= f


def perturb_wheel_inertia(model, f):
    for name in WHEEL_BODIES:
        model.body_inertia[_bid(model, name)] *= f


def perturb_chassis_inertia(model, f):
    for name in END_BODIES:
        model.body_inertia[_bid(model, name)] *= f


def perturb_spring(model, f):
    model.jnt_stiffness[_jid(model, "q_jump")] *= f


def perturb_gravity(model, f):
    model.opt.gravity[2] *= f


def perturb_tyre_friction(model, f):
    for name in ("wheel_x_geom", "wheel_y_geom"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        model.geom_friction[gid, 0] *= f


def perturb_tyre_contact(model, f):

    for name in ("wheel_x_geom", "wheel_y_geom"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        model.geom_solref[gid, 0] *= f


def perturb_end_stop_restitution(model, f):

    jid = _jid(model, "q_jump")
    field = getattr(model, "jnt_solref", None)
    if field is None:
        field = model.jnt_solreflimit
    field[jid, 1] *= f


def perturb_wheel_torque_limit(model, f):

    for name in ("tau_wheel_y", "tau_wheel_x"):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        model.actuator_ctrlrange[aid] *= f


PERTURBATIONS = {
    "total mass": perturb_mass,
    "wheel inertia": perturb_wheel_inertia,
    "chassis inertia": perturb_chassis_inertia,
    "spring stiffness": perturb_spring,
    "gravity": perturb_gravity,
    "tyre friction": perturb_tyre_friction,
    "tyre contact": perturb_tyre_contact,
    "end-stop damping": perturb_end_stop_restitution,
    "wheel torque cap": perturb_wheel_torque_limit,
}


def _final_tilt(rows):
    r = rows[-1]
    return max(abs(r["roll"]), abs(r["pitch"]))


def judge(name, rows):
    r = rows[-1]
    phase = r["phase"]
    tilt = _final_tilt(rows)
    detail = f"phase={phase} tilt={tilt:.2f} x={r['x']:.2f}"
    if name.startswith("self_right"):
        completed = phase == "balance" and tilt < 0.15
    else:
        completed = phase == "mission/complete" and tilt < 0.10 and (
            name != "jump" or r["x"] > 1.7
        )
    if completed:
        return "PASS", detail
    if tilt >= 0.30:
        return "FALL", detail
    return "INCOMPLETE", detail


def run_case(exp_name, perturb=None, factor=1.0):
    sim = Simulation()
    if perturb is not None:
        perturb(sim.model, factor)
        mujoco.mj_setConst(sim.model, sim.data)
    exp = experiments.EXPERIMENTS[exp_name]()
    try:
        rows = sim.run(exp, output=None)
    except RuntimeError as exc:
        return "ABORT", str(exc).split(":")[0]
    return judge(exp_name, rows)


def main():
    exps = ("self_right_pitch", "jump", "unified")
    levels = (0.8, 1.2)

    print("=== OFAT plant-parameter sensitivity (controllers stay nominal) ===")
    print("outcomes: PASS | ABORT (safe refusal) | INCOMPLETE (upright, unfinished) | FALL\n")
    print("baseline (nominal plant):")
    for e in exps:
        cat, detail = run_case(e)
        print(f"  {cat:10s} {e:16s} {detail}")

    print()
    header = f"{'parameter':16s} {'level':>6s}  " + "  ".join(f"{e:16s}" for e in exps)
    print(header)
    print("-" * len(header))
    for pname, pfun in PERTURBATIONS.items():
        for f in levels:
            cells = []
            for e in exps:
                try:
                    cat, _ = run_case(e, pfun, f)
                except Exception:
                    cat = "ERROR"
                cells.append(cat)
            pct = f"{int(round((f - 1) * 100)):+d}%"
            print(f"{pname:16s} {pct:>6s}  " + "  ".join(f"{c:16s}" for c in cells))


if __name__ == "__main__":
    main()
