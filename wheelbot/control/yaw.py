import math

from wheelbot.control.common import clamp, roll_pd, smoothstep, wrap_angle
from wheelbot.physical_design import DESIGN


class YawController:

    def reset(self):
        self.last_time = None
        self.filtered_rate = None
        self.lean = 0.0
        self.unloading = False
        self.wheel_y_reference = None
        self.wheel_x_reference = None
        self.requested_target = None
        self.active_target = None

    def control(self, state, target, absolute=True):
        dt = 0.0 if self.last_time is None else state.time - self.last_time
        self.last_time = state.time
        if self.filtered_rate is None:
            self.filtered_rate = state.role_yaw_rate
        elif dt > 0.0:
            fraction = dt / (0.15 + dt)
            self.filtered_rate += fraction * (
                state.role_yaw_rate - self.filtered_rate
            )

        if (
            self.requested_target is None
            or self.requested_target[1] != bool(absolute)
            or not math.isclose(
                target,
                self.requested_target[0],
                abs_tol=1e-9,
            )
        ):
            self.requested_target = float(target), bool(absolute)
            self.active_target = (
                state.role_heading
                + wrap_angle(float(target) - state.role_heading)
                if absolute
                else float(target)
            )
            self.unloading = False
            self.lean = 0.0
        error = self.active_target - state.role_heading
        direction = 1.0 if error >= 0.0 else -1.0
        remaining = abs(error)
        desired_rate = direction * min(
            0.16,
            0.50 * remaining,
            math.sqrt(2 * 0.24 * remaining),
        )
        stopping_distance = self.filtered_rate**2 / (2 * 0.24)
        braking = remaining < stopping_distance + 0.10
        if self.unloading and abs(state.balancing_wheel_speed) < 55:
            self.unloading = False
        elif not self.unloading and abs(state.balancing_wheel_speed) > 70:
            self.unloading = True

        if abs(error) < 0.12 and abs(self.filtered_rate) < 0.07:
            phase = "capture"
            self.wheel_x_reference = state.balancing_wheel_angle
            demand = clamp(0.08 * error - 0.16 * self.filtered_rate, 0.010)
            forward_speed = (
                0.004 if abs(state.balancing_wheel_speed) > 70 else 0.0
            )
        elif self.unloading:
            phase = "unload"
            fraction = min(
                1.0,
                (abs(state.balancing_wheel_speed) - 55.0) / 15.0,
            )
            fraction = smoothstep(fraction)
            rate_along_turn = direction * self.filtered_rate
            normalized = max(0.0, min(1.0, (rate_along_turn - 0.04) / (0.15 - 0.04)))
            fraction *= smoothstep(normalized)
            demand = -math.copysign(
                0.020 * fraction,
                state.balancing_wheel_speed,
            )
            forward_speed = 0.004
        elif braking:
            phase = "brake"
            demand = -direction * min(0.012, 0.040 * abs(self.filtered_rate))
            forward_speed = 0.0
        else:
            phase = "track"
            rate_error = max(
                0.0,
                direction * desired_rate - direction * self.filtered_rate,
            )
            minimum = 0.005 * min(1.0, abs(desired_rate) / 0.20)
            demand = direction * min(
                0.035,
                max(
                    minimum,
                    0.030 * abs(desired_rate) + 0.200 * rate_error,
                ),
            )
            forward_speed = 0.004

        if phase in ("unload", "capture"):
            max_rate = 0.080
        elif phase == "brake":
            max_rate = 0.040
        else:
            max_rate = 0.16 if abs(demand) < abs(self.lean) else 0.018
        self.lean += clamp(demand - self.lean, max_rate * dt)

        if self.wheel_y_reference is None:
            self.wheel_y_reference = state.driving_wheel_angle
            self.wheel_x_reference = state.balancing_wheel_angle
        target_wheel_speed = forward_speed / DESIGN.wheel_radius
        self.wheel_y_reference += target_wheel_speed * dt
        wheel_y_error = state.driving_wheel_angle - self.wheel_y_reference
        torque_y = (
            0.40 * state.driving_tilt
            + 0.04 * state.driving_tilt_rate
            + 0.006 * wheel_y_error
            + 0.007 * (state.driving_wheel_speed - target_wheel_speed)
        )
        wheel_x_error = wrap_angle(
            state.balancing_wheel_angle - self.wheel_x_reference
        )
        torque_x = (
            roll_pd(state.balancing_tilt, state.balancing_tilt_rate, self.lean)
            + 0.00008 * wheel_x_error
        )
        if phase == "capture":
            torque_x += clamp(
                -0.00003 * state.balancing_wheel_speed,
                0.010,
            )
        correction = clamp(
            0.05 * 0.8 * (desired_rate - self.filtered_rate),
            0.0004,
        )
        return (
            torque_y,
            clamp(torque_x + correction, 0.5),
            0.0,
        ), phase
