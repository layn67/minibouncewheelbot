import math


class WheelRoles:

    def reset(self):
        self.support = "wheel_y"
        self.pending = None
        self.pending_count = 0
        self.initialized = False

    def update(self, wheel_contacts, up_z):
        if not self.initialized:
            if wheel_contacts == {"wheel_x"}:
                self.support = "wheel_x"
            elif wheel_contacts == {"wheel_y"}:
                self.support = "wheel_y"
            else:
                self.support = "wheel_x" if up_z < 0.0 else "wheel_y"
            self.initialized = True

        if wheel_contacts == {"wheel_x"}:
            measured = "wheel_x"
        elif wheel_contacts == {"wheel_y"}:
            measured = "wheel_y"
        else:
            self.pending = None
            self.pending_count = 0
            return

        if measured == self.support:
            self.pending = None
            self.pending_count = 0
            return
        if measured != self.pending:
            self.pending = measured
            self.pending_count = 1
        else:
            self.pending_count += 1
        if self.pending_count >= 4:
            self.support = measured
            self.pending = None
            self.pending_count = 0

    def heading_polarity(self):
        return -1.0 if self.support == "wheel_x" else 1.0

    def direction(self, rotation):
        if self.support == "wheel_x":
            x, y = -float(rotation[0, 1]), -float(rotation[1, 1])
        else:
            x, y = float(rotation[0, 0]), float(rotation[1, 0])
        length = math.hypot(x, y)
        if length < 1e-9:
            return (
                (0.0, 1.0)
                if self.support == "wheel_x"
                else (1.0, 0.0)
            )
        return x / length, y / length

    def to_physical(self, action):
        driving, balancing, jump = action
        if self.support == "wheel_x":
            return balancing, -driving, jump
        return driving, balancing, jump

    def from_physical(self, action):
        wheel_y, wheel_x, jump = action
        if self.support == "wheel_x":
            return -wheel_x, wheel_y, jump
        return wheel_y, wheel_x, jump
