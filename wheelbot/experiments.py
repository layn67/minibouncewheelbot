import math

from wheelbot.control import UnifiedController
from wheelbot.disturbances import DisturbanceCampaign
from wheelbot.simulation import initial_position


class Experiment:
    def __init__(
        self,
        name,
        duration,
        controller,
        initial,
        control,
        disturb=None,
        obstacle=None,
    ):
        self.name = name
        self.duration = duration
        self.controller = controller
        self.initial = initial
        self.control = control
        self.disturb = disturb or (lambda model, data, state: None)
        self.obstacle = obstacle


def passive_fall():
    controller = UnifiedController()

    def control(state):
        action, phase = controller.control(state, mode="passive")
        return action, 0.0, phase

    return Experiment(
        "passive_fall",
        4.0,
        controller,
        lambda model: initial_position(model, pitch=5.0),
        control,
    )


def pitch_balance():
    controller = UnifiedController()

    def control(state):
        action, phase = controller.control(state)
        return action, 0.0, phase

    return Experiment(
        "pitch_balance",
        8.0,
        controller,
        lambda model: initial_position(model, pitch=50.0),
        control,
    )


def roll_balance():
    controller = UnifiedController()

    def control(state):
        action, phase = controller.control(state)
        return action, 0.0, phase

    return Experiment(
        "roll_balance",
        8.0,
        controller,
        lambda model: initial_position(model, roll=20.0),
        control,
    )


def motion():
    controller = UnifiedController()
    commands = (
        ("move_to", 0.45, 0.0, 0.0),
        ("move_to", 1.00, 0.0, 0.0),
        ("move_to", 0.30, 0.0, 0.0),
        ("move_to", -0.35, 0.0, 0.0),
        ("move_to", 0.0, 0.0, 0.0),
    )

    def control(state):
        action, phase = controller.control(
            state,
            mode="mission",
            commands=commands,
        )
        index = min(controller.command_index, len(commands) - 1)
        return action, commands[index][1], phase

    return Experiment(
        "motion",
        60.0,
        controller,
        lambda model: initial_position(model),
        control,
    )


def disturbance():
    controller = UnifiedController()
    campaign = DisturbanceCampaign()

    def control(state):
        action, phase = controller.control(state)
        return action, 0.0, f"rejection/{phase}"

    return Experiment(
        "disturbance",
        campaign.duration,
        controller,
        lambda model: initial_position(model),
        control,
        disturb=campaign.apply,
    )


def yaw():
    controller = UnifiedController()
    commands = (("yaw", math.pi), ("yaw", 0.0))

    def control(state):
        action, phase = controller.control(
            state,
            mode="mission",
            commands=commands,
        )
        index = min(controller.command_index, len(commands) - 1)
        target = commands[index][1]
        return action, target, phase

    return Experiment(
        "yaw",
        120.0,
        controller,
        lambda model: initial_position(model),
        control,
    )


def yaw_wheel_x():
    controller = UnifiedController()
    commands = (("yaw", math.pi), ("yaw", 0.0))

    def control(state):
        action, phase = controller.control(
            state,
            mode="mission",
            commands=commands,
        )
        index = min(controller.command_index, len(commands) - 1)
        return action, commands[index][1], phase

    return Experiment(
        "yaw_wheel_x",
        120.0,
        controller,
        lambda model: initial_position(model, roll=180.0),
        control,
    )


def self_right_pitch():
    controller = UnifiedController()

    def control(state):
        action, phase = controller.control(state)
        return action, 0.0, phase

    return Experiment(
        "self_right_pitch",
        24.0,
        controller,
        lambda model: initial_position(model, pitch=80.0),
        control,
    )


def self_right_roll():
    controller = UnifiedController()

    def control(state):
        action, phase = controller.control(state)
        return action, 0.0, phase

    return Experiment(
        "self_right_roll",
        24.0,
        controller,
        lambda model: initial_position(model, roll=100.0),
        control,
    )


def jump():
    features = (
        (0.70, 0.024, 0.026),
        (1.60, 0.036, 0.035),
    )
    destination = 1.90
    controller = UnifiedController()
    commands = tuple(
        (
            ("jump", feature, destination, 0.0)
            if index == len(features) - 1
            else ("jump", feature, None, 0.0)
        )
        for index, feature in enumerate(features)
    )

    def control(state):
        action, phase = controller.control(
            state,
            mode="mission",
            commands=commands,
        )
        return action, destination, phase

    return Experiment(
        "jump",
        30.0,
        controller,
        lambda model: initial_position(model),
        control,
        obstacle=features,
    )


def unified():
    features = ((1.18, 0.024, 0.026),)
    controller = UnifiedController()
    commands = (
        ("move_to", 0.0, 0.0, math.pi / 2),
        ("yaw", 0.0),
        ("move_to", 0.20, 0.0, 0.0),
        ("jump", features[0], 1.50, 0.0),
    )

    def control(state):
        action, phase = controller.control(
            state,
            mode="mission",
            commands=commands,
        )
        return action, 1.18, phase

    return Experiment(
        "unified",
        120.0,
        controller,
        lambda model: initial_position(
            model,
            pitch=80.0,
            yaw=90.0,
            y=-0.50,
        ),
        control,
        obstacle=features,
    )


EXPERIMENTS = {
    "passive_fall": passive_fall,
    "pitch_balance": pitch_balance,
    "roll_balance": roll_balance,
    "motion": motion,
    "disturbance": disturbance,
    "yaw": yaw,
    "yaw_wheel_x": yaw_wheel_x,
    "self_right_pitch": self_right_pitch,
    "self_right_roll": self_right_roll,
    "jump": jump,
    "unified": unified,
}
