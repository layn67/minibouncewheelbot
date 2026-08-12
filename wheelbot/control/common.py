import math

ZERO = (0.0, 0.0, 0.0)


def clamp(value, limit):
    return max(-limit, min(limit, float(value)))


def wrap_angle(angle):
    return math.remainder(angle, 2.0 * math.pi)


def smoothstep(x):
    return x * x * (3.0 - 2.0 * x)


def heading_vector(heading):
    return math.cos(heading), math.sin(heading)


ROLL_STIFFNESS = 1.30
ROLL_DAMPING = 0.16


def roll_pd(tilt, tilt_rate, target=0.0, damping=ROLL_DAMPING):
    return ROLL_STIFFNESS * (tilt - target) + damping * tilt_rate


def has_fallen(state):
    tilt = max(abs(state.driving_tilt), abs(state.balancing_tilt))
    return tilt > 0.90 and (abs(state.up_z) < 0.45 or state.z < 0.055)


def blend_near_upright(angle):
    return max(0.0, min(1.0, (0.14 - abs(angle)) / 0.08))


def catch_wheel_blend(angle):
    return max(0.0, min(1.0, (0.66 - abs(angle)) / 0.01))
