import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalDesign:

    gravity: float = 9.81
    end_body_mass: float = 0.215
    wheel_mass: float = 0.129
    wheel_radius: float = 0.032
    wheel_offset: float = 0.066
    wheel_axial_inertia: float = 120e-6
    spring_stiffness: float = 6000.0
    spring_stroke: float = 0.030
    jump_force_limit: float = 120.0
    wheel_torque_limit: float = 0.5
    motor_base_speed_rpm: float = 7000.0
    motor_no_load_speed_rpm: float = 8000.0

    @staticmethod
    def _rpm_to_rad_s(rpm):
        return rpm * 2.0 * math.pi / 60.0

    @property
    def motor_base_speed(self):

        return self._rpm_to_rad_s(self.motor_base_speed_rpm)

    @property
    def motor_no_load_speed(self):

        return self._rpm_to_rad_s(self.motor_no_load_speed_rpm)

    @property
    def reaction_wheel_max_speed(self):

        return self.motor_no_load_speed

    @property
    def total_mass(self):

        return 2.0 * (self.end_body_mass + self.wheel_mass)

    @property
    def upright_height(self):

        return self.wheel_offset + self.wheel_radius

    @property
    def maximum_static_preload(self):

        return min(
            self.spring_stroke,
            self.jump_force_limit / self.spring_stiffness,
        )


DESIGN = PhysicalDesign()
