import math

from wheelbot.physical_design import DESIGN

_CHASSIS_BODIES = (
    ("wheel_x_end", 0.215, +0.014, 0.000179),
    ("wheel_y_end", 0.215, -0.014, 0.000179),
    ("wheel_x", 0.129, +0.066, 0.000120),
    ("wheel_y", 0.129, -0.066, 0.000063),
)


def chassis_transverse_inertia():

    return sum(
        inertia + mass * offset**2
        for _, mass, offset, inertia in _CHASSIS_BODIES
    )


BARRIER_RADIUS = 0.051
BARRIER_BODY_INERTIA = 0.00175
BARRIER_ENERGY_RESERVE = 1.20
BARRIER_TRANSFER_EFFICIENCY = 0.32


def barrier_wheel_speed(height, design=DESIGN):

    mass = design.total_mass
    barrier = max(0.0, BARRIER_RADIUS - height)
    energy = BARRIER_ENERGY_RESERVE * mass * design.gravity * barrier
    support_inertia = BARRIER_BODY_INERTIA + mass * BARRIER_RADIUS**2
    body_momentum = math.sqrt(2.0 * support_inertia * energy)
    return body_momentum / (BARRIER_TRANSFER_EFFICIENCY * design.wheel_axial_inertia)
