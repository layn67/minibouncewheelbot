import math

from wheelbot.control.common import blend_near_upright, clamp, roll_pd, wrap_angle
from wheelbot.physical_design import DESIGN


class BalanceController:

    def reset(self):
        self.last_time = None
        self.engagement_start = None
        self.position_integral = 0.0
        self.wheel_x_reference = None
        self.heading_reference = None

    def control(
        self,
        state,
        x_target=0.0,
        vx_target=0.0,
        roll_target=0.0,
        heading_target=None,
        path_direction=None,
        speed_limit=0.50,
    ):
        if self.wheel_x_reference is None:
            self.wheel_x_reference = state.balancing_wheel_angle

        dt = 0.0 if self.last_time is None else state.time - self.last_time
        self.last_time = state.time
        if self.engagement_start is None:
            self.engagement_start = state.time
        engagement = min(
            1.0,
            max(0.0, (state.time - self.engagement_start) / 0.25),
        )
        if path_direction is None:
            measured_position = state.drive_position
            measured_path_speed = state.drive_velocity
            role_projection = 1.0
        else:
            path_x, path_y = path_direction
            measured_position = path_x * state.x + path_y * state.y
            measured_path_speed = path_x * state.vx + path_y * state.vy
            role_projection = (
                state.drive_direction[0] * path_x
                + state.drive_direction[1] * path_y
            )
        position_error = measured_position - x_target
        pitch_blend = blend_near_upright(state.driving_tilt)
        if pitch_blend > 0.5 and engagement >= 1.0:
            self.position_integral = clamp(
                self.position_integral + position_error * dt, 0.5
            )
        elif pitch_blend < 0.1:
            self.position_integral = 0.0
        commanded_path_speed = clamp(
            vx_target
            - 0.50 * position_error
            - 0.10 * self.position_integral
            - 0.18 * (measured_path_speed - vx_target),
            speed_limit,
        )
        commanded_speed = commanded_path_speed * role_projection
        pitch_target = clamp(
            -0.42 * (state.drive_velocity - commanded_speed),
            0.20,
        )
        pitch_target *= pitch_blend * engagement
        torque_y = (
            0.40 * (state.driving_tilt - pitch_target)
            + 0.05 * state.driving_tilt_rate
        )
        torque_y += clamp(
            DESIGN.wheel_axial_inertia
            * state.balancing_wheel_speed
            * state.role_yaw_rate,
            0.10,
        )

        roll_blend = blend_near_upright(state.balancing_tilt)
        wheel_x_error = wrap_angle(
            state.balancing_wheel_angle - self.wheel_x_reference
        )
        momentum_torque = roll_blend * engagement * clamp(
            0.00008 * wheel_x_error + 0.00040 * state.balancing_wheel_speed,
            0.020,
        )
        if self.heading_reference is None:
            self.heading_reference = (
                state.yaw
                if heading_target is None
                else heading_target
            )
        elif heading_target is not None:
            self.heading_reference = heading_target
        heading_error = wrap_angle(self.heading_reference - state.yaw)
        role_heading_error = state.heading_polarity * heading_error
        steering_speed = 0.08
        if abs(commanded_speed) >= steering_speed:
            heading_authority = math.copysign(1.0, commanded_speed)
        elif abs(state.drive_velocity) >= steering_speed:
            heading_authority = math.copysign(1.0, state.drive_velocity)
        else:
            heading_authority = 0.0
        heading_tilt = (
            roll_blend
            * engagement
            * clamp(
                heading_authority
                * (
                    0.060 * role_heading_error
                    - 0.025 * state.role_yaw_rate
                ),
                0.008,
            )
        )
        balancing_target = roll_target + heading_tilt
        torque_x = (
            roll_pd(state.balancing_tilt, state.balancing_tilt_rate, balancing_target)
            + momentum_torque
        )
        return clamp(torque_y, 0.5), clamp(torque_x, 0.5), 0.0
