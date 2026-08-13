import math

from wheelbot.control.common import (
    ZERO,
    catch_wheel_blend,
    clamp,
    has_fallen,
    roll_pd,
)
from wheelbot.physical_design import DESIGN
from wheelbot.self_right_model import barrier_wheel_speed


class SelfRightController:

    def reset(self):
        self.axis = None
        self.phase = "settle"
        self.settle_start = None
        self.settled_since = None
        self.settle_contact = None
        self.settle_face = None
        self.direction = None
        self.pitch_direction = None
        self.failed_since = None
        self.retry_axis = None
        self.best_up_z = None
        self.progress_time = None

    def roll_angle(self, state):
        return math.atan2(state.body_up_y, state.body_up_z)

    def pitch_angle(self, state):
        return math.atan2(-state.body_up_x, state.body_up_z)

    def fallen(self, state):
        return has_fallen(state)

    def _dominant_face(self, state):
        pitch = abs(state.body_up_x)
        roll = abs(state.body_up_y)
        return max(pitch, roll) >= 0.98 and abs(pitch - roll) >= 0.75

    def _choose_face(self, state):
        pitch = abs(state.body_up_x)
        roll = abs(state.body_up_y)
        if pitch > roll:
            return "pitch"
        if roll > pitch:
            return "roll"
        return (
            "pitch"
            if abs(self.pitch_angle(state)) >= abs(self.roll_angle(state))
            else "roll"
        )

    def _settle_action(self, state):
        roll = self.roll_angle(state)
        pitch = self.pitch_angle(state)
        torque_y = -0.0004 * state.wheel_y_speed
        torque_x = -0.0004 * state.wheel_x_speed
        if self.settle_face is None:
            self.settle_face = self._choose_face(state)
        if state.grounded and not self._dominant_face(state):
            if self.settle_face == "pitch":
                torque_x += 0.03 * roll + 0.01 * state.raw_roll_rate
            else:
                torque_y += 0.03 * pitch + 0.01 * state.raw_pitch_rate
        return clamp(torque_y, 0.08), clamp(torque_x, 0.08), 0.0

    def _settled(self, state):
        if self.settle_start is None:
            self.settle_start = state.time
        elapsed = state.time - self.settle_start
        linear_speed = max(abs(state.vx), abs(state.vy), abs(state.vz))
        angular_rate = max(
            abs(state.raw_roll_rate),
            abs(state.raw_pitch_rate),
            abs(state.yaw_rate),
        )
        wheel_speed = max(abs(state.wheel_x_speed), abs(state.wheel_y_speed))
        nominal = (
            elapsed >= 0.25
            and state.grounded
            and self._dominant_face(state)
            and linear_speed < 0.05
            and angular_rate < 0.30
            and wheel_speed < 100.0
        )
        timeout = (
            elapsed >= 4.0
            and state.grounded
            and linear_speed < 1.20
            and angular_rate < 18.0
            and wheel_speed < 100.0
        )
        contact = (
            state.wheel_x_grounded,
            state.wheel_y_grounded,
            state.support_wheel,
        )
        if contact != self.settle_contact:
            self.settle_contact = contact
            self.settled_since = None
            nominal = False
        if nominal:
            if self.settled_since is None:
                self.settled_since = state.time
        else:
            self.settled_since = None
        return timeout or (
            self.settled_since is not None
            and state.time - self.settled_since >= 0.25
        )

    def _select_axis(self, state):
        body_pitch = abs(state.body_up_x)
        body_roll = abs(state.body_up_y)
        roll = abs(self.roll_angle(state))
        pitch = abs(self.pitch_angle(state))
        if max(body_pitch, body_roll) > 0.15:
            if body_pitch >= body_roll + 0.12:
                return "pitch"
            if body_roll >= body_pitch + 0.12:
                return "roll"
            return "pitch" if pitch >= roll else "roll"
        if max(roll, pitch) > 0.65:
            return "pitch" if pitch >= roll else "roll"
        return self._choose_face(state)

    def _catch_ready(self, state, axis):
        angle = (
            self.roll_angle(state)
            if axis == "roll"
            else self.pitch_angle(state)
        )
        limit = 0.65 if axis == "roll" else 0.50
        return state.up_z > 0.80 and abs(angle) < limit

    def _release_ready(self, state):
        return (
            self.phase == "catch"
            and abs(self.roll_angle(state)) < 0.30
            and abs(self.pitch_angle(state)) < 0.30
            and state.up_z > 0.90
            and state.z > 0.052
            and math.sqrt(
                state.raw_roll_rate**2
                + state.raw_pitch_rate**2
                + state.yaw_rate**2
            )
            < 3.0
            and math.hypot(state.vx, state.vy) < 0.50
            and abs(state.vz) < 0.35
            and max(abs(state.wheel_x_speed), abs(state.wheel_y_speed)) < 100
        )

    def _barrier_momentum_speed(self, state):
        return barrier_wheel_speed(state.z)

    def _retry(self, state):
        alternate = "roll" if self.axis == "pitch" else "pitch"
        self.reset()
        self.retry_axis = alternate
        self.settle_face = alternate
        self.settle_start = state.time
        return self._settle_action(state), self.phase

    def control(self, state):
        if self.phase == "balance":
            if not self.fallen(state):
                return ZERO, self.phase
            self.reset()

        if self.phase == "catch" and self.fallen(state):
            body_quiet = (
                max(abs(state.vx), abs(state.vy), abs(state.vz)) < 0.05
                and max(
                    abs(state.raw_roll_rate),
                    abs(state.raw_pitch_rate),
                    abs(state.yaw_rate),
                )
                < 0.30
            )
            if body_quiet:
                if self.failed_since is None:
                    self.failed_since = state.time
                elif state.time - self.failed_since >= 0.50:
                    return self._retry(state)
            else:
                self.failed_since = None
        else:
            self.failed_since = None

        if self.phase not in ("settle", "balance") and self.fallen(state):
            if (
                self.best_up_z is None
                or state.up_z >= self.best_up_z + 0.02
            ):
                self.best_up_z = state.up_z
                self.progress_time = state.time
            elif (
                self.progress_time is not None
                and state.time - self.progress_time >= 6.0
            ):
                return self._retry(state)

        if self.phase == "settle":
            if not self._settled(state):
                return self._settle_action(state), self.phase
            self.axis = self.retry_axis or self._select_axis(state)
            self.best_up_z = state.up_z
            self.progress_time = state.time
            if self.axis == "roll":
                self.direction = -1.0 if state.body_up_y > 0.0 else 1.0
                self.phase = "rock" if state.up_z >= 0.30 else "drive"
            else:
                self.direction = -1.0 if state.body_up_x > 0.0 else 1.0
                self.pitch_direction = (
                    1.0 if self.pitch_angle(state) >= 0.0 else -1.0
                )
                self.phase = "charge" if state.wheel_y_grounded else "drive"

        roll = self.roll_angle(state)
        pitch = self.pitch_angle(state)

        if self._release_ready(state):
            self.phase = "balance"
            return ZERO, self.phase

        if self.axis == "roll":
            direction = self.direction or 1.0
            if self.phase == "rock" and abs(roll) >= 1.40:
                self.phase = "rock_coast"
            if (
                self.phase == "rock_coast"
                and direction * state.raw_roll_rate >= 0.0
            ):
                self.phase = "launch"
            if self.phase == "drive" and self._catch_ready(state, "roll"):
                self.phase = "catch"
            elif self.phase == "drive" and abs(state.wheel_x_speed) >= min(
                DESIGN.reaction_wheel_max_speed,
                self._barrier_momentum_speed(state),
            ):
                self.phase = "launch"
            if self.phase == "launch":
                transfer_complete = (
                    -direction * state.wheel_x_speed >= 400.0
                    and abs(roll) <= 0.65
                    and direction * state.raw_roll_rate >= 0.0
                )
                if transfer_complete:
                    self.phase = "coast"
                elif self._catch_ready(state, "roll"):
                    self.phase = "catch"
            if self.phase == "coast" and self._catch_ready(state, "roll"):
                self.phase = "catch"

            if self.phase == "rock":
                torque_x = direction * 0.10
            elif self.phase in ("rock_coast", "coast"):
                torque_x = 0.0
            elif self.phase == "drive":
                torque_x = direction * min(0.5, 0.05 * abs(roll))
            elif self.phase == "launch":
                torque_x = -direction * 0.5
            else:
                torque_x = clamp(
                    roll_pd(roll, state.roll_rate)
                    + catch_wheel_blend(roll)
                    * 0.0004
                    * state.wheel_x_speed,
                    0.5,
                )
            torque_y = clamp(0.02 * state.raw_pitch_rate, 0.5)
        else:
            tilt_direction = self.pitch_direction or 1.0
            recovery_direction = -(self.direction or 1.0)
            if self.phase == "drive" and self._catch_ready(state, "pitch"):
                self.phase = "catch"
            elif self.phase == "drive":
                speed_limit = min(
                    DESIGN.reaction_wheel_max_speed,
                    self._barrier_momentum_speed(state),
                )
                if abs(state.wheel_y_speed) >= speed_limit:
                    self.phase = "launch"
            if self.phase == "charge" and (
                tilt_direction * state.wheel_y_speed <= -100.0
            ):
                self.phase = "climb"
            if self.phase == "launch":
                transfer_complete = (
                    -tilt_direction * state.wheel_y_speed <= 5.0
                    and -tilt_direction * state.raw_pitch_rate > 0.0
                )
                if transfer_complete:
                    self.phase = "coast"
                elif state.wheel_y_grounded:
                    self.phase = "climb"
                elif self._catch_ready(state, "pitch"):
                    self.phase = "catch"
            if self.phase == "climb" and self._catch_ready(state, "pitch"):
                self.phase = "catch"
            if self.phase == "coast" and self._catch_ready(state, "pitch"):
                self.phase = "catch"

            if self.phase == "charge":
                torque_y = -tilt_direction * 0.05
            elif self.phase == "drive":
                torque_y = recovery_direction * (
                    0.05 * min(abs(pitch), math.pi / 2)
                ) + 0.03 * state.raw_pitch_rate
                torque_y = clamp(torque_y, 0.5)
            elif self.phase == "launch":
                torque_y = clamp(-recovery_direction * 0.5, 0.5)
            elif self.phase == "coast":
                torque_y = 0.0
            else:
                torque_y = clamp(
                    1.50 * pitch
                    + 0.16 * state.raw_pitch_rate
                    + 0.50 * state.body_vx,
                    0.5,
                )
            torque_x = (
                clamp(
                    roll_pd(roll, state.raw_roll_rate, damping=0.40)
                    + catch_wheel_blend(roll)
                    * 0.0004
                    * state.wheel_x_speed,
                    0.5,
                )
                if self.phase == "catch"
                else 0.0
            )

        return (torque_y, torque_x, 0.0), self.phase
