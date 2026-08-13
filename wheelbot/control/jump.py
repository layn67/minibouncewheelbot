import math

from wheelbot.control.common import clamp, heading_vector
from wheelbot.jump_planner import plan_jump
from wheelbot.physical_design import DESIGN


class JumpController:

    def reset(self):
        self.phase = "approach"
        self.cleared = False
        self.lifted_off = False
        self.integral = 0.0
        self.last_time = None
        self.speed_reference = None
        self.ready_since = None
        self.complete_since = None
        self.recovery_position = None
        self.planned_feature = None
        self.plan = None

    def _plan(self, feature):
        feature = tuple(map(float, feature))
        if feature == self.planned_feature:
            return dict(self.plan)
        center, width, _ = feature
        plan = plan_jump(feature)
        plan["center"] = center
        plan["end"] = center + width / 2
        plan["speed"] = plan["horizontal"]
        plan["release"] = center - plan["speed"] * (
            plan["crossing_time"] + plan["release_delay"]
        )
        plan["preload_start"] = plan["release"] - 0.18
        self.planned_feature = feature
        self.plan = dict(plan)
        return plan

    def _slew_speed(self, target, dt):
        if self.speed_reference is None:
            self.speed_reference = 0.0
        maximum_change = 0.40 * dt
        self.speed_reference += clamp(
            target - self.speed_reference,
            maximum_change,
        )
        return self.speed_reference

    def control(self, state, feature, destination=None, heading=0.0):
        if feature is None or len(feature) != 3:
            raise ValueError("a jump command requires one terrain feature")
        dt = 0.0 if self.last_time is None else state.time - self.last_time
        self.last_time = state.time
        path_direction = heading_vector(heading)
        position = path_direction[0] * state.x + path_direction[1] * state.y
        path_velocity = (
            path_direction[0] * state.vx
            + path_direction[1] * state.vy
        )

        feature_end = float(feature[0]) + float(feature[1]) / 2
        if (
            not self.cleared
            and self.phase == "approach"
            and position > feature_end + 0.04
        ):
            raise RuntimeError(
                "jump rejected: the referenced terrain feature is behind "
                "the robot"
            )

        if self.cleared:
            if destination is None:
                return (
                    (0.0, 0.0, 0.0),
                    position,
                    0.0,
                    "complete",
                    False,
                    True,
                )
            calm = (
                state.grounded
                and abs(position - float(destination)) < 0.04
                and abs(path_velocity) < 0.18
                and max(
                    abs(state.driving_tilt),
                    abs(state.balancing_tilt),
                )
                < 0.08
                and max(
                    abs(state.driving_tilt_rate),
                    abs(state.balancing_tilt_rate),
                )
                < 0.10
                and abs(state.role_yaw_rate) < 0.05
                and max(
                    abs(state.driving_wheel_speed),
                    abs(state.balancing_wheel_speed),
                )
                < 20.0
            )
            if calm:
                if self.complete_since is None:
                    self.complete_since = state.time
            else:
                self.complete_since = None
            complete = (
                self.complete_since is not None
                and state.time - self.complete_since >= 0.20
            )
            speed = (
                clamp(2.0 * (float(destination) - position), 0.65)
                if destination is not None
                else 0.0
            )
            return (
                (0.0, 0.0, 0.0),
                position,
                speed,
                "complete" if complete else "post_stop",
                False,
                complete,
            )

        plan = self._plan(feature)
        measured_speed = max(0.05, min(plan["maximum_horizontal"], path_velocity))
        plan["release"] = plan["center"] - measured_speed * (
            plan["crossing_time"] + plan["release_delay"]
        )
        plan["preload_start"] = plan["release"] - 0.18
        support_ready = (
            state.grounded
            and max(
                abs(state.driving_tilt),
                abs(state.balancing_tilt),
            )
            < 0.12
            and max(
                abs(state.driving_tilt_rate),
                abs(state.balancing_tilt_rate),
            )
            < 0.75
            and abs(state.vz) < 0.35
            and abs(state.role_yaw_rate) < 0.30
            and abs(state.balancing_wheel_speed) < 20.0
        )
        speed_ready = (
            path_velocity >= plan["minimum_horizontal"]
            and path_velocity <= plan["maximum_horizontal"]
        )

        if self.phase == "approach" and position >= plan["preload_start"]:
            self.phase = "preload"
        if (
            self.phase == "preload"
            and position >= plan["release"] - 0.003
            and state.jump >= 0.98 * plan["preload"]
            and support_ready
            and speed_ready
        ):
            self.phase = "release"
        if (
            self.phase in ("approach", "preload")
            and position >= plan["clearance_entry"]
        ):
            raise RuntimeError(
                "jump aborted: the tyre reached the obstacle envelope "
                "before launch conditions were feasible"
            )
        if self.phase == "release" and not state.grounded:
            self.phase = "flight"
            self.lifted_off = True
        if (
            self.phase in ("release", "flight")
            and self.lifted_off
            and state.grounded
        ):
            self.phase = "recover"
            self.ready_since = None
            self.recovery_position = position
        if self.phase == "recover":
            recovered = (
                support_ready
                and abs(state.vz) < 0.35
                and math.hypot(state.vx, state.vy) < 0.65
                and max(
                    abs(state.driving_wheel_speed),
                    abs(state.balancing_wheel_speed),
                )
                < 20.0
                and position > plan["end"] + 0.04
            )
            if recovered:
                if self.ready_since is None:
                    self.ready_since = state.time
            else:
                self.ready_since = None
            if (
                self.ready_since is not None
                and state.time - self.ready_since >= 0.05
            ):
                self.cleared = True
                self.phase = "post_stop"
                self.complete_since = None

        executing = self.phase in ("release", "flight")
        if executing:
            action = (
                clamp(
                    0.40 * state.driving_tilt
                    + 0.05 * state.driving_tilt_rate,
                    0.28,
                ),
                clamp(
                    0.40 * state.balancing_tilt
                    + 0.05 * state.balancing_tilt_rate,
                    0.28,
                ),
                0.0,
            )
            return action, position, 0.0, self.phase, True, False

        speed = self._slew_speed(plan["speed"], dt)
        force = 0.0
        if self.phase == "preload":
            progress = max(
                0.0,
                min(
                    1.0,
                    (position - plan["preload_start"])
                    / max(1e-6, plan["release"] - plan["preload_start"]),
                ),
            )
            extension_target = plan["preload"] * progress
            extension_speed = (
                plan["preload"]
                * max(0.0, path_velocity)
                / max(1e-6, plan["release"] - plan["preload_start"])
                if 0.0 < progress < 1.0
                else 0.0
            )
            error = extension_target - state.jump
            self.integral = clamp(self.integral + error * dt, 0.02)
            force = (
                DESIGN.spring_stiffness * extension_target
                + 1000.0 * error
                + 80.0 * self.integral
                + 12.0 * (extension_speed - state.jump_speed)
            )
        elif self.phase == "recover":
            speed = 0.0
        support_position = (
            self.recovery_position
            if self.phase == "recover" and self.recovery_position is not None
            else position
        )
        return (
            (0.0, 0.0, clamp(force, DESIGN.jump_force_limit)),
            support_position,
            speed,
            self.phase,
            False,
            False,
        )
