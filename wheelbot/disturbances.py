import numpy as np


class DisturbanceCampaign:

    pulse_duration = 0.12
    duration = 40.0

    labels = ("forward push (pitch)", "backward push (pitch)",
              "side push, right (roll)", "side push, left (roll)",
              "hard shove: knockdown → self-right")
    events = (
        (3.0, "force", (2.0, 0.0, 0.0)),
        (9.0, "force", (-2.0, 0.0, 0.0)),
        (15.0, "force", (0.0, 0.8, 0.0)),
        (21.0, "force", (0.0, -0.8, 0.0)),
        (27.0, "impulse", (0, 3.5)),
    )

    def __init__(self):
        self.applied = set()

    def apply(self, model, data, state):
        body_id = model.body("wheelbot").id
        rotation = data.xmat[body_id].reshape(3, 3)
        for index, (start, kind, spec) in enumerate(self.events):
            if kind == "impulse":
                if state.time >= start and index not in self.applied:
                    data.qvel[spec[0]] += spec[1]
                    self.applied.add(index)
            elif start <= state.time < start + self.pulse_duration:
                if kind == "force":
                    data.xfrc_applied[body_id, :3] = rotation @ np.asarray(spec)
                else:
                    data.xfrc_applied[body_id, 3:] = rotation @ np.asarray(spec)
