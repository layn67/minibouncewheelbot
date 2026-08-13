import math

from wheelbot.control.balance import BalanceController
from wheelbot.control.common import ZERO, has_fallen, heading_vector, wrap_angle
from wheelbot.control.jump import JumpController
from wheelbot.control.self_right import SelfRightController
from wheelbot.control.yaw import YawController


class UnifiedController:

    def __init__(self):
        self.balance = BalanceController()
        self.yaw = YawController()
        self.self_right = SelfRightController()
        self.jump = JumpController()

    def reset(self):
        self.balance.reset()
        self.yaw.reset()
        self.self_right.reset()
        self.jump.reset()
        self.active_mode = None
        self.recovering = False
        self.command_index = 0
        self.active_command = None
        self.command_target = None
        self.command_start = None
        self.command_ready_since = None
        self.jump_executing = False
        self.jump_recovery_heading = None
        self.recovery_support_engaged = False
        self.recovery_position = None
        self.recovery_heading = None
        self.mission_commands = None
        self.pending_commands = None
        self.completion_position = None
        self.completion_direction = None
        self.completion_heading = None
        self.idle_position = None
        self.idle_heading = None
        self.standalone_jump_feature = None
        self.standalone_jump_heading = None

    def _select(self, mode):
        if mode == self.active_mode:
            return
        if mode == "balance":
            self.balance.reset()
            self.idle_position = None
            self.idle_heading = None
        elif mode == "yaw":
            self.yaw.reset()
        elif mode == "jump":
            self.jump.reset()
            self.balance.reset()
            self.jump_executing = False
            self.jump_recovery_heading = None
            self.standalone_jump_feature = None
            self.standalone_jump_heading = None
        self.active_mode = mode

    def control(
        self,
        state,
        mode="balance",
        x_target=None,
        vx_target=0.0,
        roll_target=0.0,
        yaw_target=0.0,
        heading_target=None,
        commands=None,
        feature=None,
        destination=None,
    ):
        if mode == "passive":
            return ZERO, "passive"

        if self.recovering or has_fallen(state):
            if not self.recovering:
                self.self_right.reset()
                self.recovering = True
                self.active_mode = None
                self.recovery_support_engaged = False
            action, phase = self.self_right.control(state)
            if phase == "balance":
                if not self.recovery_support_engaged:
                    self.balance.reset()
                    self.recovery_support_engaged = True
                    self.recovery_position = state.drive_position
                    self.recovery_heading = state.yaw
                action = self.balance.control(
                    state,
                    x_target=self.recovery_position,
                    vx_target=0.0,
                    heading_target=self.recovery_heading,
                )
            else:
                action = state.roles.from_physical(action)
            if (
                phase == "balance"
                and state.up_z > 0.90
                and max(abs(state.roll), abs(state.pitch)) < 0.25
            ):
                self.recovering = False
                self.active_mode = "balance"
                self.idle_position = self.recovery_position
                self.idle_heading = self.recovery_heading
            return action, f"recover/{phase}"

        if mode == "mission":
            if commands is None:
                raise ValueError("mission mode requires a command sequence")
            return self.run_sequence(state, commands)
        self._select(mode)
        if mode == "balance":
            if x_target is None:
                if self.idle_position is None:
                    self.idle_position = state.drive_position
                    self.idle_heading = state.yaw
                x_target = self.idle_position
                if heading_target is None:
                    heading_target = self.idle_heading
            return self.balance.control(
                state,
                x_target=x_target,
                vx_target=vx_target,
                roll_target=roll_target,
                heading_target=heading_target,
            ), "balance"
        if mode == "yaw":
            role_target = state.heading_polarity * yaw_target
            return self.yaw.control(state, role_target)
        if mode == "jump":
            resolved_feature = (
                None if feature is None else tuple(map(float, feature))
            )
            if resolved_feature != self.standalone_jump_feature:
                self.jump.reset()
                self.balance.reset()
                self.jump_executing = False
                self.jump_recovery_heading = None
                self.standalone_jump_feature = resolved_feature
                self.standalone_jump_heading = (
                    state.yaw
                    if heading_target is None
                    else float(heading_target)
                )
            action, phase, _ = self._control_jump(
                state,
                feature,
                destination,
                self.standalone_jump_heading,
            )
            return action, phase
        raise ValueError(f"unknown control mode: {mode}")

    def _next_command(self):
        self.command_index += 1
        self.active_command = None
        self.command_target = None
        self.command_start = None
        self.command_ready_since = None
        self.active_mode = None

    def _motion_complete(
        self,
        state,
        position_error,
        speed,
        heading_error,
    ):
        return (
            abs(position_error) < 0.040
            and abs(speed) < 0.18
            and max(
                abs(state.driving_tilt),
                abs(state.balancing_tilt),
            )
            < 0.12
            and max(
                abs(state.driving_tilt_rate),
                abs(state.balancing_tilt_rate),
                abs(state.role_yaw_rate),
            )
            < 1.5
            and abs(heading_error) < 0.12
            and abs(state.driving_wheel_speed) < 20.0
            and abs(state.balancing_wheel_speed) < 5.0
        )

    def _control_jump(self, state, feature, destination, heading):
        if heading is None:
            heading = state.yaw
        (
            jump_action,
            support_position,
            support_velocity,
            phase,
            executing,
            complete,
        ) = self.jump.control(state, feature, destination, heading)
        if executing:
            self.jump_executing = True
            return jump_action, phase, complete
        if self.jump_executing:
            self.balance.reset()
            self.jump_executing = False
            self.jump_recovery_heading = state.yaw
        hold_heading = (
            self.jump_recovery_heading
            if self.jump_recovery_heading is not None
            else heading
        )
        path_direction = heading_vector(hold_heading)
        support = self.balance.control(
            state,
            x_target=support_position,
            vx_target=support_velocity,
            heading_target=hold_heading,
            path_direction=path_direction,
            speed_limit=max(0.50, abs(support_velocity)),
        )
        action = support[0], support[1], jump_action[2]
        return action, phase, complete

    def run_sequence(self, state, commands):
        requested_commands = tuple(commands)
        if self.mission_commands is None:
            self.mission_commands = requested_commands
        elif requested_commands != self.mission_commands:
            current = self.active_command
            matching_index = None
            if current is not None:
                if (
                    self.command_index < len(requested_commands)
                    and requested_commands[self.command_index] == current
                ):
                    matching_index = self.command_index
                else:
                    try:
                        matching_index = requested_commands.index(current)
                    except ValueError:
                        pass
            if matching_index is not None:
                self.mission_commands = requested_commands
                self.command_index = matching_index
            elif (
                self.active_mode == "jump"
                and not self.jump.cleared
                and (
                    self.jump.phase != "approach"
                    or state.jump > 0.001
                )
            ):
                self.pending_commands = requested_commands
            else:
                self.mission_commands = requested_commands
                self.pending_commands = None
                self.command_index = 0
                self.active_command = None
                self.command_target = None
                self.command_start = None
                self.command_ready_since = None
                self.active_mode = None
                self.completion_position = None
                self.completion_direction = None
                self.completion_heading = None
        if (
            self.pending_commands is not None
            and (
                self.active_mode != "jump"
                or self.jump.cleared
                or (
                    self.jump.phase == "approach"
                    and state.jump <= 0.001
                )
            )
        ):
            self.mission_commands = self.pending_commands
            self.pending_commands = None
            self.command_index = 0
            self.active_command = None
            self.command_target = None
            self.command_start = None
            self.command_ready_since = None
            self.active_mode = None
            self.completion_position = None
            self.completion_direction = None
            self.completion_heading = None

        commands = self.mission_commands
        if self.command_index >= len(commands):
            self._select("balance")
            if self.completion_position is None:
                self.completion_position = (state.x, state.y)
                self.completion_heading = state.yaw
                self.completion_direction = heading_vector(state.yaw)
            path_x, path_y = self.completion_direction
            target = (
                path_x * self.completion_position[0]
                + path_y * self.completion_position[1]
            )
            action = self.balance.control(
                state,
                x_target=target,
                heading_target=self.completion_heading,
                path_direction=self.completion_direction,
            )
            return action, "mission/complete"

        command = commands[self.command_index]
        if command != self.active_command:
            self.active_command = command
            self.command_target = None
            self.command_start = None
            self.command_ready_since = None
            self.active_mode = None
        name = command[0]
        label = f"mission/{self.command_index + 1}_{name}"

        if name == "move":
            self._select("balance")
            if self.command_target is None:
                self.command_target = state.drive_position + float(command[1])
            action = self.balance.control(
                state,
                x_target=self.command_target,
            )
            heading = self.balance.heading_reference
            heading_error = wrap_angle(state.yaw - heading)
            if self._motion_complete(
                state,
                state.drive_position - self.command_target,
                state.drive_velocity,
                heading_error,
            ):
                self._next_command()
            return action, label

        if name == "move_to":
            self._select("balance")
            x_target, y_target, heading = map(float, command[1:4])
            path_direction = heading_vector(heading)
            travel_target = (
                path_direction[0] * x_target
                + path_direction[1] * y_target
            )
            action = self.balance.control(
                state,
                x_target=travel_target,
                heading_target=heading,
                path_direction=path_direction,
            )
            measured_position = (
                path_direction[0] * state.x
                + path_direction[1] * state.y
            )
            measured_speed = (
                path_direction[0] * state.vx
                + path_direction[1] * state.vy
            )
            if self._motion_complete(
                state,
                measured_position - travel_target,
                measured_speed,
                wrap_angle(state.yaw - heading),
            ):
                self._next_command()
            return action, label

        if name in ("yaw", "turn"):
            self._select("yaw")
            if self.command_target is None:
                requested = float(command[1])
                self.command_target = (
                    state.yaw + requested
                    if name == "turn"
                    else state.yaw
                    + wrap_angle(requested - state.yaw)
                )
            role_target = state.heading_polarity * self.command_target
            action, phase = self.yaw.control(
                state,
                role_target,
                absolute=name == "yaw",
            )
            ready = (
                abs(state.yaw - self.command_target) <= 0.060
                and abs(state.role_yaw_rate) <= 0.070
                and abs(state.balancing_tilt) <= 0.025
                and abs(state.balancing_tilt_rate) <= 0.080
                and abs(state.balancing_wheel_speed) <= 50.0
            )
            if ready:
                if self.command_ready_since is None:
                    self.command_ready_since = state.time
            else:
                self.command_ready_since = None
            if (
                self.command_ready_since is not None
                and state.time - self.command_ready_since >= 0.10
            ):
                self._next_command()
            return action, f"{label}/{phase}"

        if name == "jump":
            self._select("jump")
            if len(command) < 2:
                raise ValueError("jump command requires terrain reference data")
            feature = command[1]
            destination = command[2] if len(command) >= 3 else None
            if self.command_target is None:
                self.jump.reset()
                self.jump_recovery_heading = None
                self.command_target = True
                self.command_start = (
                    float(command[3])
                    if len(command) >= 4
                    else state.yaw
                )
            action, phase, complete = self._control_jump(
                state,
                feature,
                destination,
                self.command_start,
            )
            if complete:
                self._next_command()
            return action, f"{label}/{phase}"

        raise ValueError(f"unknown sequence command: {name}")
