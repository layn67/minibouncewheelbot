from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wheelbot.physical_design import DESIGN
from wheelbot.self_right_model import (
    BARRIER_BODY_INERTIA,
    BARRIER_ENERGY_RESERVE,
    BARRIER_RADIUS,
    BARRIER_TRANSFER_EFFICIENCY,
    chassis_transverse_inertia,
)


_ROOT_Z = DESIGN.upright_height
_WHEEL_OFFSET = DESIGN.wheel_offset

_END_BODY_INERTIA_ROLL = 0.000179
_SUPPORT_WHEEL_INERTIA_ROLL = 0.000063


@dataclass(frozen=True)
class LinearModel:

    name: str
    states: tuple[str, ...]
    inputs: tuple[str, ...]
    A: np.ndarray
    B: np.ndarray
    params: dict[str, float]

    @property
    def poles(self) -> np.ndarray:
        return np.linalg.eigvals(self.A)

    @property
    def unstable_poles(self) -> np.ndarray:
        p = self.poles
        return p[p.real > 1e-9]

    def controllability_matrix(self) -> np.ndarray:
        n = self.A.shape[0]
        cols = [self.B]
        for _ in range(1, n):
            cols.append(self.A @ cols[-1])
        return np.hstack(cols)

    @property
    def is_controllable(self) -> bool:
        return np.linalg.matrix_rank(self.controllability_matrix()) == self.A.shape[0]


def _roll_lumped_parameters() -> dict[str, float]:

    g = DESIGN.gravity
    m_end = DESIGN.end_body_mass
    m_wheel = DESIGN.wheel_mass
    I_w = DESIGN.wheel_axial_inertia

    parts = [
        ("support_wheel_y", m_wheel, _ROOT_Z - _WHEEL_OFFSET, _SUPPORT_WHEEL_INERTIA_ROLL),
        ("wheel_y_end",     m_end,   _ROOT_Z - 0.014,        _END_BODY_INERTIA_ROLL),
        ("wheel_x_end",     m_end,   _ROOT_Z + 0.014,        _END_BODY_INERTIA_ROLL),
        ("reaction_wheel_x", m_wheel, _ROOT_Z + _WHEEL_OFFSET, 0.0),
    ]

    total_mass = sum(m for _, m, _, _ in parts)
    P = sum(m * h for _, m, h, _ in parts)
    J_p = sum(m * h * h + I for _, m, h, I in parts)
    com_height = P / total_mass

    return {
        "g": g,
        "P_first_moment": P,
        "J_p": J_p,
        "I_w": I_w,
        "total_mass": total_mass,
        "com_height": com_height,
    }


def roll_reaction_wheel_model() -> LinearModel:

    p = _roll_lumped_parameters()
    Pg = p["P_first_moment"] * p["g"]
    J_p = p["J_p"]
    I_w = p["I_w"]

    A = np.array([
        [0.0,      1.0, 0.0],
        [Pg / J_p, 0.0, 0.0],
        [0.0,      0.0, 0.0],
    ])
    B = np.array([
        [0.0],
        [-1.0 / J_p],
        [1.0 / I_w],
    ])

    return LinearModel(
        name="roll_reaction_wheel_pendulum",
        states=("phi", "phi_dot", "theta_dot"),
        inputs=("tau",),
        A=A,
        B=B,
        params=p,
    )


_WHEEL_PITCH_INERTIA = 0.000120
_END_BODY_INERTIA_PITCH = 0.000179
_WHEEL_X_INERTIA_PITCH = 0.000063


def _pitch_lumped_parameters() -> dict[str, float]:

    g = DESIGN.gravity
    r_w = DESIGN.wheel_radius
    m_w = DESIGN.wheel_mass
    I_w = _WHEEL_PITCH_INERTIA

    axle = DESIGN.wheel_radius
    m_end = DESIGN.end_body_mass
    m_wheel = DESIGN.wheel_mass

    parts = [
        ("wheel_y_end", m_end,   (_ROOT_Z - 0.014) - axle, _END_BODY_INERTIA_PITCH),
        ("wheel_x_end", m_end,   (_ROOT_Z + 0.014) - axle, _END_BODY_INERTIA_PITCH),
        ("wheel_x",     m_wheel, (_ROOT_Z + _WHEEL_OFFSET) - axle, _WHEEL_X_INERTIA_PITCH),
    ]

    m_b = sum(m for _, m, _, _ in parts)
    l = sum(m * h for _, m, h, _ in parts) / m_b
    I_b = sum(I + m * (h - l) ** 2 for _, m, h, I in parts)
    J_psi = I_b + m_b * l * l

    M_t = m_w + I_w / (r_w * r_w) + m_b
    p = m_b * l
    G = m_b * g * l
    D = M_t * J_psi - p * p

    return {
        "g": g, "r_w": r_w,
        "m_b": m_b, "l": l, "I_b": I_b, "J_psi": J_psi,
        "M_t": M_t, "p_coupling": p, "G_gravity": G, "det_D": D,
    }


def pitch_wheeled_pendulum_model() -> LinearModel:

    q = _pitch_lumped_parameters()
    r_w, p, G, D, J_psi, M_t = (q["r_w"], q["p_coupling"], q["G_gravity"],
                                q["det_D"], q["J_psi"], q["M_t"])
    A = np.array([
        [0.0, 1.0, 0.0,          0.0],
        [0.0, 0.0, -p * G / D,   0.0],
        [0.0, 0.0, 0.0,          1.0],
        [0.0, 0.0, M_t * G / D,  0.0],
    ])
    B = np.array([
        [0.0],
        [(J_psi / r_w + p) / D],
        [0.0],
        [-(p / r_w + M_t) / D],
    ])
    return LinearModel("pitch_wheeled_inverted_pendulum",
                       ("pos", "pos_dot", "psi", "psi_dot"), ("tau",), A, B, q)


ROLL_GAINS = {
    "k_phi": 1.30,
    "k_phidot": 0.16,
    "k_theta": 0.00008,
    "k_thetadot": 0.00040,
}


def roll_model_full() -> LinearModel:
    p = _roll_lumped_parameters()
    Pg = p["P_first_moment"] * p["g"]
    J_p = p["J_p"]
    I_w = p["I_w"]
    A = np.array([
        [0.0,      1.0, 0.0, 0.0],
        [Pg / J_p, 0.0, 0.0, 0.0],
        [0.0,      0.0, 0.0, 1.0],
        [0.0,      0.0, 0.0, 0.0],
    ])
    B = np.array([[0.0], [-1.0 / J_p], [0.0], [1.0 / I_w]])
    return LinearModel("roll_full_with_wheel_angle",
                       ("phi", "phi_dot", "theta", "theta_dot"),
                       ("tau",), A, B, p)


def roll_closed_loop(gains: dict[str, float] = ROLL_GAINS) -> None:
    m = roll_model_full()
    K = np.array([[gains["k_phi"], gains["k_phidot"],
                   gains["k_theta"], gains["k_thetadot"]]])
    print("\n=== Roll closed-loop audit (balance.py gains) ===")
    print("gains K =", K)
    for label, Acl in (("IMPLEMENTED  A + B K  (tau = +K x)", m.A + m.B @ K),
                       ("counterfactual A - B K (not implemented)",
                        m.A - m.B @ K)):
        poles = np.linalg.eigvals(Acl)
        stable = np.all(poles.real < 1e-9)
        print(f"  {label}: poles = "
              f"{np.array2string(poles, precision=3, suppress_small=True)}"
              f"  -> {'STABLE' if stable else 'UNSTABLE'}")


def yaw_steering_model(forward_speed: float, steering_constant: float = 10.0) -> LinearModel:

    p = _roll_lumped_parameters()
    Pg = p["P_first_moment"] * p["g"]
    J_p = p["J_p"]
    c_v = steering_constant * forward_speed
    A = np.array([
        [0.0,      1.0, 0.0],
        [Pg / J_p, 0.0, 0.0],
        [c_v,      0.0, 0.0],
    ])
    B = np.array([[0.0], [-1.0 / J_p], [0.0]])
    params = dict(p, forward_speed=forward_speed, steering_constant=steering_constant)
    return LinearModel("yaw_through_lean", ("phi", "phi_dot", "psi"), ("tau",),
                       A, B, params)


def yaw_controllability_demo() -> None:

    print("\n=== Yaw controllability vs forward speed ===")
    print("state = [phi, phi_dot, psi];  yaw couples as psi_dot = c*v*phi")
    print("(c is a structural placeholder, not identified -- the rank result")
    print(" depends only on whether c*v vanishes; see yaw_authority_budget)")
    for v in (0.0, 0.05, 0.30):
        m = yaw_steering_model(forward_speed=v)
        rank = np.linalg.matrix_rank(m.controllability_matrix())
        verdict = "controllable" if m.is_controllable else "UNCONTROLLABLE (yaw unreachable)"
        print(f"  v = {v:.2f} m/s : controllability rank {rank}/3  -> {verdict}")


def jump_energy_model(efficiency: float = 0.425) -> dict[str, float]:

    k = DESIGN.spring_stiffness
    F = DESIGN.jump_force_limit
    stroke = DESIGN.spring_stroke
    M = DESIGN.total_mass
    g = DESIGN.gravity

    x_max = min(stroke, F / k)
    E_spring = 0.5 * k * x_max ** 2
    E_launch = efficiency * E_spring
    v_launch = np.sqrt(2.0 * E_launch / M)
    apex = E_launch / (M * g)
    flight_time = 2.0 * v_launch / g

    return {
        "k": k, "F": F, "M": M, "efficiency": efficiency,
        "x_max_preload": x_max,
        "force_limited": F / k < stroke,
        "E_spring_max": E_spring,
        "E_launch": E_launch,
        "v_launch": v_launch,
        "apex_height": apex,
        "flight_time": flight_time,
    }


def end_stop_launch_model(preload, design=DESIGN):

    k = design.spring_stiffness
    g = design.gravity
    M = design.total_mass
    m_s = 0.5 * M
    x = float(preload)
    stored = 0.5 * k * x * x
    sprung_kinetic = stored - m_s * g * x
    v_sprung = np.sqrt(max(0.0, 2.0 * sprung_kinetic / m_s))
    v_cg = (m_s / M) * v_sprung
    eta = (m_s / M) * (1.0 - 2.0 * m_s * g / (k * x)) if x > 0 else 0.0
    return {
        "preload": x,
        "stored_energy": stored,
        "m_sprung": m_s,
        "v_sprung": v_sprung,
        "v_takeoff": v_cg,
        "eta_ideal": eta,
        "structural_ceiling": m_s / M,
    }


def end_stop_launch_report() -> None:
    from wheelbot.jump_planner import IdentifiedLaunchModel

    identified = IdentifiedLaunchModel().retained_energy_lower_bound
    print("\n=== End-stop launch: ideal coefficient vs identified value ===")
    print("  the grounded half never leaves the floor during spring extension")
    print("  (N = M g + k q > 0); take-off is the internal end-stop collision,")
    print("  so v_cg is set by the sprung half's momentum at q = 0.")
    print(f"  structural ceiling m_sprung/M = "
          f"{end_stop_launch_model(0.020)['structural_ceiling']:.3f} "
          f"(before any dissipation)")
    print(f"  {'preload(mm)':>11s} {'v_sprung':>9s} {'v_takeoff':>10s} "
          f"{'eta_ideal':>10s} {'two-mass':>9s}")
    for x in (0.012, 0.016, 0.020):
        m = end_stop_launch_model(x)
        tm = two_mass_launch_model(x)
        print(f"  {x*1000:11.0f} {m['v_sprung']:9.3f} {m['v_takeoff']:10.3f} "
              f"{m['eta_ideal']:10.3f} {tm['eta_derived']:9.3f}")
    print(f"  identified from the plant (jump_validation): {identified:.3f}")
    print("  The identified value is a chassis-ROOT apex coefficient, measured")
    print("  in the same frame the planner and the clearance test use, so the")
    print("  pipeline is self-consistent; it is not literally an energy")
    print("  conversion efficiency, and should not be reported as one.")


def self_right_barrier_model() -> None:
    m = DESIGN.total_mass
    g = DESIGN.gravity
    I_w = DESIGN.wheel_axial_inertia
    h0 = DESIGN.upright_height

    radius_c, body_inertia_c = BARRIER_RADIUS, BARRIER_BODY_INERTIA
    reserve, transfer = BARRIER_ENERGY_RESERVE, BARRIER_TRANSFER_EFFICIENCY
    cap = DESIGN.reaction_wheel_max_speed

    radius_derived = h0 / 2.0
    body_inertia_exact = chassis_transverse_inertia()
    I_edge = body_inertia_c + m * radius_c ** 2

    print("\n=== Self-righting: energy-barrier momentum-transfer model ===")
    print(f"  controller radius {radius_c} m  vs  plant half-height h0/2 = {radius_derived:.3f} m")
    print(f"  controller body_inertia {body_inertia_c}  vs  exact four-body "
          f"parallel-axis sum = {body_inertia_exact:.7f} kg m^2 "
          f"(error {100 * abs(body_inertia_c - body_inertia_exact) / body_inertia_exact:.2f}%)")
    print(f"  body inertia about tipping edge I_edge = {I_edge:.5f} kg m^2")
    print(f"  reserve = {reserve} (20% energy margin), transfer eta = {transfer} "
          f"(wheel->body momentum coefficient: TUNED, not identified --")
    print("   self_right_analysis.py estimates it a posteriori from logged runs)")
    print(f"  {'CoM rise dh(mm)':>15s} {'barrier E(J)':>12s} {'omega_wheel(rad/s)':>18s}")
    for dh in (0.010, 0.020, 0.030):
        E = reserve * m * g * dh
        L_body = np.sqrt(2.0 * I_edge * E)
        omega = L_body / (transfer * I_w)
        print(f"  {dh*1000:15.0f} {E:12.3f} {omega:18.0f}")
    print(f"  (controller caps the pitch spin-up at {cap:.0f} rad/s, so tall barriers")
    print("   saturate the wheel authority -- the physical limit of the flip.)")

    spin_up, transfer_to = 425.0, -400.0
    delta_omega = spin_up - transfer_to
    L_body = transfer * I_w * delta_omega
    E_cross = L_body ** 2 / (2.0 * I_edge)
    dh_cross = E_cross / (reserve * m * g)
    print("  --- roll axis: barrier-derived entry, fixed 400 rad/s exit ---")
    print(f"  a {spin_up:.0f} -> {transfer_to:.0f} rad/s wheel swing imparts "
          f"L_body = {L_body:.4f} kg m^2/s at the assumed eta, which crosses a")
    print(f"  ~{dh_cross*1000:.0f} mm barrier -- comparable to the pitch ceiling below,")
    print("  which is why the fixed exit test and the derived entry threshold are")
    print("  mutually consistent for this plant. The rock phase (rock to")
    print("  |roll| = 1.40 rad, then flip) is amplitude pumping, used when the")
    print("  chassis starts partially recovered rather than flat.")


def yaw_authority_budget() -> None:

    p = _roll_lumped_parameters()
    Pg = p["P_first_moment"] * p["g"]
    I_w = DESIGN.wheel_axial_inertia
    I_yaw = 2 * 0.000131 + 0.000063 + 0.000063
    tau_w = DESIGN.wheel_torque_limit
    omega_unload = 70.0
    omega_release = 55.0
    max_lean = 0.035
    balance_envelope = 0.14

    print("\n=== Yaw authority: what actually limits the turn ===")
    print(f"  body yaw inertia I_yaw = {I_yaw:.2e} kg m^2 "
          f"(all mass lies on the yaw axis by symmetry)")
    print("  (1) at standstill: momentum along gravity is conserved under any")
    print("      internal wheel torque -> yaw is NOT commandable from rest")
    print("      (Muehlebach and D'Andrea, 2017; Gajamohan et al., 2013).")
    print("  (2) vertical projection of the wheel momentum while tilted:")
    for tilt, label in ((max_lean, "yaw.py lean ceiling"),
                        (balance_envelope, "balancing envelope")):
        Lz = I_w * omega_unload * np.sin(tilt)
        print(f"      tilt {tilt:.3f} rad ({label:19s}): "
              f"L_z = {Lz:.2e} kg m^2/s -> equivalent yaw rate "
              f"{Lz / I_yaw:5.2f} rad/s")
    print("  (3) lean-hold budget (the actual constraint):")
    for lean in (0.012, 0.020, max_lean):
        wheel_accel = Pg * lean / I_w
        hold_time = (omega_unload - omega_release) / wheel_accel
        spin_time = omega_unload / wheel_accel
        print(f"      lean {lean:.3f} rad -> wheel ramps at {wheel_accel:6.1f} "
              f"rad/s^2; {spin_time:.2f} s from rest to the {omega_unload:.0f} "
              f"rad/s unload threshold, {hold_time:.2f} s per track/unload cycle")
    print(f"  wheel torque is NOT binding: holding the {max_lean:.3f} rad lean "
          f"needs {Pg * max_lean:.4f} N m of {tau_w:.2f} N m available "
          f"({tau_w / (Pg * max_lean):.0f}x margin).")
    print("  => the in-place turn is momentum-duty-cycle limited. A coordinated")
    print("     arc turn, which converts lean into path curvature while moving,")
    print("     is the model-identified route to faster continuous turning.")


def actuator_envelope_report() -> None:

    tau = DESIGN.wheel_torque_limit
    base = DESIGN.motor_base_speed
    no_load = DESIGN.motor_no_load_speed
    mass = DESIGN.total_mass
    peak = tau * base

    print("\n=== Wheel-actuator envelope: implied peak mechanical power ===")
    print(f"  peak torque {tau:.2f} N m held flat to base speed "
          f"{base:.0f} rad/s ({DESIGN.motor_base_speed_rpm:.0f} rpm)")
    print(f"  implied peak mechanical power = {peak:.0f} W per wheel "
          f"= {peak / mass:.0f} W/kg of total robot mass")
    print(f"  derates linearly to zero at the no-load speed "
          f"{no_load:.0f} rad/s ({DESIGN.motor_no_load_speed_rpm:.0f} rpm)")
    print("  For scale, Haldane et al. (2016) calculate in a design study")
    print("  that a hypothetical 0.25 kg rigidly-actuated robot would need")
    print("  343 W/kg to match a galago's 1.74 m jump, against 90 W/kg for a")
    print("  series-elastic actuator with a mechanical-advantage profile;")
    print("  their built prototype Salto measures 137 W/kg. Those are peak")
    print("  burst figures for a jump, not sustained availability, and the")
    print("  comparison is a power-density sanity check on the actuator")
    print("  model rather than a claim about jumping. The envelope here is")
    print("  therefore optimistic and must be declared as such: no power,")
    print("  current or thermal limit is enforced by the plant, so results at")
    print("  sustained high wheel speed (the self-right spin-up in particular)")
    print("  should be read as an upper bound on achievable performance.")


def jump_preload_closed_loop() -> None:
    k = DESIGN.spring_stiffness
    c = 1.0772
    m = 0.344 * 0.344 / (0.344 + 0.344)
    Kp, Ki, Kd = 1000.0, 80.0, 12.0

    wn0 = np.sqrt(k / m)
    z0 = c / (2.0 * np.sqrt(k * m))
    wn = np.sqrt((k + Kp) / m)
    z = (c + Kd) / (2.0 * np.sqrt((k + Kp) * m))
    A = np.array([
        [0.0,             1.0,             0.0],
        [-(k + Kp) / m,  -(c + Kd) / m,   -Ki / m],
        [1.0,             0.0,             0.0],
    ])
    poles = np.linalg.eigvals(A)
    stable = np.all(poles.real < 1e-9)

    print("\n=== Preload-servo (q_jump) gain justification ===")
    print(f"  spring alone (no control): wn={wn0:.0f} rad/s ({wn0/(2*np.pi):.0f} Hz), "
          f"zeta={z0:.3f}  (near-undamped -> rings)")
    print(f"  with PID feedforward     : wn={wn:.0f} rad/s ({wn/(2*np.pi):.0f} Hz), "
          f"zeta={z:.3f}")
    print(f"  closed-loop poles: {np.array2string(poles, precision=1, suppress_small=True)}"
          f"  -> {'STABLE' if stable else 'UNSTABLE'}")
    print(f"  Kd raises damping zeta {z0:.3f} -> {z:.3f} (its job: damp the near-undamped spring);")
    print(f"  Kp stiffens; Ki nulls steady-state extension error; feedforward k*x_ref holds preload.")


def two_mass_launch_model(preload, m_sprung=0.344, m_unsprung=0.344):
    k = DESIGN.spring_stiffness
    g = DESIGN.gravity
    m_B, m_F = m_sprung, m_unsprung
    m_T = m_B + m_F
    d = preload
    kd = k * d
    numerator = kd * (kd - 2.0 * m_B * g) - m_F * g * g * (2.0 * m_B - m_F)
    v_cg = (m_B / m_T) * np.sqrt(max(0.0, numerator / (k * m_B)))
    epe = 0.5 * k * d * d
    ke = 0.5 * m_T * v_cg * v_cg
    eta = ke / epe if epe > 0 else 0.0
    return {"v_takeoff": v_cg, "eta_derived": eta,
            "m_sprung": m_B, "m_unsprung": m_F, "preload": d}


def two_mass_efficiency_report() -> None:
    from wheelbot.jump_planner import IdentifiedLaunchModel
    fitted = IdentifiedLaunchModel().retained_energy_lower_bound
    print("\n=== Derived launch efficiency (Lo & Parslew two-mass model) ===")
    print(f"  sim-identified (fitted) eta = {fitted:.3f}  [planner uses this]")
    print(f"  {'preload(mm)':>11s} {'v_takeoff':>10s} {'eta_derived':>12s}")
    for x in (0.012, 0.016, 0.020):
        m = two_mass_launch_model(x)
        print(f"  {x*1000:11.0f} {m['v_takeoff']:10.3f} {m['eta_derived']:12.3f}")
    print("  Note: derived eta is the IDEAL (no damping); the sim value is lower")
    print("  by the guide/contact damping absent from the analytical model.")


def jump_energy_report() -> None:
    m = jump_energy_model()
    print("\n=== Jump energy model (1-DOF spring -> ballistic) ===")
    for k, v in m.items():
        print(f"  {k:16s} = {v:.5g}")
    g, M, eff, ks = (DESIGN.gravity, DESIGN.total_mass, m["efficiency"],
                     DESIGN.spring_stiffness)
    print("  --- minimum preload to reach a target apex ---")
    for h in (0.026, 0.042, m["apex_height"]):
        x = np.sqrt(2.0 * M * g * h / (eff * ks))
        feasible = x <= m["x_max_preload"]
        print(f"    apex {h*1000:5.1f} mm -> preload {x*1000:5.1f} mm  "
              f"({'feasible' if feasible else 'INFEASIBLE'})")


def pitch_closed_loop() -> None:

    m = pitch_wheeled_pendulum_model()

    k_x = 0.40 * 0.21
    k_v = 0.40 * 0.4956
    k_psi = 0.40
    k_psidot = 0.05
    K = np.array([[k_x, k_v, k_psi, k_psidot]])

    print("\n=== Pitch closed-loop audit (balance.py cascade) ===")
    print("effective K = [x, v, psi, psi_dot] =",
          np.array2string(K, precision=4, suppress_small=True))
    for label, Acl in (("IMPLEMENTED  A + B K  (tau = +K x)", m.A + m.B @ K),
                       ("counterfactual A - B K (not implemented)",
                        m.A - m.B @ K)):
        poles = np.linalg.eigvals(Acl)
        stable = np.all(poles.real < 1e-9)
        print(f"  {label}: poles = "
              f"{np.array2string(poles, precision=3, suppress_small=True)}"
              f"  -> {'STABLE' if stable else 'UNSTABLE'}")


def _report(model: LinearModel) -> None:
    print(f"=== {model.name} ===")
    print("Lumped physical parameters:")
    for k, v in model.params.items():
        print(f"  {k:16s} = {v:.6g}")
    print("\nA =\n", np.array2string(model.A, precision=4, suppress_small=True))
    print("B^T =", np.array2string(model.B.T, precision=4, suppress_small=True))
    poles = model.poles
    print("\nOpen-loop poles:", np.array2string(poles, precision=4, suppress_small=True))
    unstable = model.unstable_poles
    if unstable.size:
        tau_fall = 1.0 / unstable.real.max()
        print(f"  -> UNSTABLE: RHP pole at {unstable.real.max():.3f} rad/s "
              f"(fall time constant ~{tau_fall*1000:.0f} ms)")
    print(f"Controllable: {model.is_controllable} "
          f"(rank {np.linalg.matrix_rank(model.controllability_matrix())}/{model.A.shape[0]})")


if __name__ == "__main__":
    _report(roll_reaction_wheel_model())
    roll_closed_loop()
    print()
    _report(pitch_wheeled_pendulum_model())
    pitch_closed_loop()
    yaw_controllability_demo()
    yaw_authority_budget()
    actuator_envelope_report()
    self_right_barrier_model()
    jump_energy_report()
    end_stop_launch_report()
    jump_preload_closed_loop()
    two_mass_efficiency_report()
