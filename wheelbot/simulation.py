from pathlib import Path
import csv
import math
import time

import mujoco
import numpy as np

from wheelbot.control.common import clamp, wrap_angle
from wheelbot.physical_design import DESIGN
from wheelbot.roles import WheelRoles


MODEL_PATH = Path(__file__).parent / "model" / "wheelbot.xml"


def quaternion(roll_degrees=0.0, pitch_degrees=0.0, yaw_degrees=0.0):
    roll = math.radians(roll_degrees)
    pitch = math.radians(pitch_degrees)
    yaw = math.radians(yaw_degrees)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def rotation_matrix(quat):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def lowest_collision_point(model, data, geom_id):
    geom_type = int(model.geom_type[geom_id])
    center_z = float(data.geom_xpos[geom_id, 2])
    size = model.geom_size[geom_id]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)

    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return center_z - float(np.dot(np.abs(rotation[2]), size))
    if geom_type in (
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
    ):
        axis_z = float(rotation[2, 2])
        radial_z = math.sqrt(max(0.0, 1.0 - axis_z * axis_z))
        return center_z - float(size[1]) * abs(axis_z) - float(size[0]) * radial_z
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return center_z - float(size[0])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        end = start + int(model.mesh_vertnum[mesh_id])
        return center_z + float(np.min(model.mesh_vert[start:end] @ rotation[2]))
    return center_z - float(model.geom_rbound[geom_id])


def contact_height(model, qpos):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    lowest = math.inf
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name == "floor":
            continue
        if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
            continue
        lowest = min(lowest, lowest_collision_point(model, data, geom_id))
    return max(0.0, -lowest)


def initial_position(
    model,
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
    x=0.0,
    y=0.0,
    z=None,
):
    qpos = np.zeros(model.nq)
    qpos[:3] = (x, y, 0.0)
    qpos[3:7] = quaternion(roll, pitch, yaw)
    if z is None:
        qpos[2] = contact_height(model, qpos)
    else:
        qpos[2] = z
    return qpos


class State:

    def __init__(self, simulation):
        model = simulation.model
        data = simulation.data
        self.roles = simulation.roles
        rotation = rotation_matrix(data.xquat[simulation.base_body])
        up = rotation[:, 2]
        body_up = rotation[2, :]

        velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_XBODY,
            simulation.base_body,
            velocity,
            0,
        )
        world_angular = velocity[:3]
        world_linear = velocity[3:]
        body_angular = rotation.T @ world_angular
        body_linear = rotation.T @ world_linear

        self.time = float(data.time)
        self.x, self.y, self.z = map(float, data.xpos[simulation.base_body])
        self.vx, self.vy, self.vz = map(float, world_linear)
        self.body_vx = float(body_linear[0])
        raw_roll = math.atan2(rotation[2, 1], rotation[2, 2])
        raw_pitch = math.asin(max(-1.0, min(1.0, -rotation[2, 0])))
        self.roll = raw_roll
        self.pitch = raw_pitch
        self.raw_roll = raw_roll
        self.raw_pitch = raw_pitch
        self.raw_roll_rate = float(body_angular[0])
        self.raw_pitch_rate = float(body_angular[1])
        self.roll_rate = float(body_angular[0])
        self.pitch_rate = float(body_angular[1])
        cos_pitch = math.cos(raw_pitch)
        self.yaw_rate = (
            (
                math.sin(raw_roll) * body_angular[1]
                + math.cos(raw_roll) * body_angular[2]
            )
            / cos_pitch
            if abs(cos_pitch) > 1e-6
            else float(body_angular[2])
        )
        self.up_x, self.up_y, self.up_z = map(float, up)
        self.body_up_x, self.body_up_y, self.body_up_z = map(float, body_up)

        raw_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        self.yaw = simulation.unwrap_yaw(raw_yaw)
        wheel_contacts = simulation.wheel_contacts()
        self.wheel_x_grounded = "wheel_x" in wheel_contacts
        self.wheel_y_grounded = "wheel_y" in wheel_contacts
        simulation.roles.update(wheel_contacts, self.up_z)
        self.supported_on_wheel = (
            wheel_contacts == {simulation.roles.support}
            and self.z > 0.075
        )
        self.support_wheel = simulation.roles.support
        self.heading_polarity = simulation.roles.heading_polarity()
        self.role_heading = self.heading_polarity * self.yaw
        self.role_yaw_rate = self.heading_polarity * self.yaw_rate

        self.wheel_y_angle = float(data.qpos[simulation.wheel_y_qpos])
        self.wheel_x_angle = float(data.qpos[simulation.wheel_x_qpos])
        self.wheel_y_speed = float(data.qvel[simulation.wheel_y_qvel])
        self.wheel_x_speed = float(data.qvel[simulation.wheel_x_qvel])
        if self.support_wheel == "wheel_x":
            self.driving_tilt = -wrap_angle(raw_roll - math.pi)
            self.driving_tilt_rate = -self.raw_roll_rate
            self.balancing_tilt = -raw_pitch
            self.balancing_tilt_rate = self.raw_pitch_rate
            self.driving_wheel_angle = -self.wheel_x_angle
            self.driving_wheel_speed = -self.wheel_x_speed
            self.balancing_wheel_angle = self.wheel_y_angle
            self.balancing_wheel_speed = self.wheel_y_speed
        else:
            self.driving_tilt = raw_pitch
            self.driving_tilt_rate = self.raw_pitch_rate
            self.balancing_tilt = raw_roll
            self.balancing_tilt_rate = self.raw_roll_rate
            self.driving_wheel_angle = self.wheel_y_angle
            self.driving_wheel_speed = self.wheel_y_speed
            self.balancing_wheel_angle = self.wheel_x_angle
            self.balancing_wheel_speed = self.wheel_x_speed

        self.drive_direction = simulation.roles.direction(rotation)
        direction = np.asarray(self.drive_direction)
        self.drive_position = float(direction @ data.xpos[simulation.base_body, :2])
        self.drive_velocity = float(direction @ world_linear[:2])
        self.jump = float(data.qpos[simulation.jump_qpos])
        self.jump_speed = float(data.qvel[simulation.jump_qvel])
        self.grounded = simulation.has_floor_contact()

    def row(self, target, phase, role_control, physical_control):
        return {
            "time": self.time,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "vx": self.vx,
            "vy": self.vy,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "roll_rate": self.roll_rate,
            "pitch_rate": self.pitch_rate,
            "yaw_rate": self.yaw_rate,
            "wheel_x_speed": self.wheel_x_speed,
            "wheel_y_speed": self.wheel_y_speed,
            "jump_extension": self.jump,
            "jump_speed": self.jump_speed,
            "target": target,
            "phase": phase,
            "driving_torque": role_control[0],
            "balancing_torque": role_control[1],
            "torque_y": physical_control[0],
            "torque_x": physical_control[1],
            "jump_force": physical_control[2],
            "support_wheel": self.support_wheel,
            "grounded": int(self.grounded),
        }


class Simulation:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.base_body = self.body("wheelbot")
        self.wheel_y_qpos, self.wheel_y_qvel = self.joint_addresses("q_wheel_y")
        self.wheel_x_qpos, self.wheel_x_qvel = self.joint_addresses("q_wheel_x")
        self.jump_qpos, self.jump_qvel = self.joint_addresses("q_jump")
        self.actuator_y = self.actuator("tau_wheel_y")
        self.actuator_x = self.actuator("tau_wheel_x")
        self.actuator_jump = self.actuator("tau_jump")
        self.floor_geom = self.geom("floor")
        self.wheel_x_geom = self.geom("wheel_x_geom")
        self.wheel_y_geom = self.geom("wheel_y_geom")
        self.roles = WheelRoles()
        self.raw_yaw = None
        self.continuous_yaw = 0.0
        self.roles.reset()

    def body(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def geom(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def actuator(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def joint_addresses(self, name):
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_qposadr[joint]), int(self.model.jnt_dofadr[joint])

    def reset(self, qpos):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.raw_yaw = None
        self.continuous_yaw = 0.0
        mujoco.mj_forward(self.model, self.data)

    def unwrap_yaw(self, raw_yaw):
        if self.raw_yaw is None:
            self.continuous_yaw = raw_yaw
        else:
            self.continuous_yaw += wrap_angle(raw_yaw - self.raw_yaw)
        self.raw_yaw = raw_yaw
        return self.continuous_yaw

    def has_floor_contact(self):
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.floor_geom in (contact.geom1, contact.geom2):
                return True
        return False

    def wheel_contacts(self):
        contacts = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {contact.geom1, contact.geom2}
            if self.floor_geom not in pair:
                continue
            if self.wheel_x_geom in pair:
                contacts.add("wheel_x")
            if self.wheel_y_geom in pair:
                contacts.add("wheel_y")
        return contacts

    def place_obstacle(
        self,
        x=0.75,
        height=0.035,
        half_width=0.025,
        index=1,
        y=0.0,
        heading=0.0,
    ):
        geom = self.geom(f"demo_feature_{index}")
        body = int(self.model.geom_bodyid[geom])
        mocap = int(self.model.body_mocapid[body])
        self.model.geom_size[geom] = (half_width, 0.18, height / 2)
        self.model.geom_contype[geom] = 1
        self.model.geom_conaffinity[geom] = 1
        self.model.geom_rgba[geom, 3] = 1
        self.data.mocap_pos[mocap] = (x, y, height / 2)
        self.data.mocap_quat[mocap] = quaternion(
            yaw_degrees=math.degrees(heading)
        )
        mujoco.mj_forward(self.model, self.data)

    def apply(self, physical_control):
        self.data.ctrl[self.actuator_y] = self.wheel_torque(
            physical_control[0],
            self.data.qvel[self.wheel_y_qvel],
        )
        self.data.ctrl[self.actuator_x] = self.wheel_torque(
            physical_control[1],
            self.data.qvel[self.wheel_x_qvel],
        )
        self.data.ctrl[self.actuator_jump] = clamp(
            physical_control[2],
            DESIGN.jump_force_limit,
        )

    def wheel_torque(self, command, speed):
        torque = clamp(command, DESIGN.wheel_torque_limit)
        if torque * speed <= 0.0:
            return torque
        base_speed = DESIGN.motor_base_speed
        maximum_speed = DESIGN.motor_no_load_speed
        if abs(speed) <= base_speed:
            return torque
        authority = max(
            0.0,
            (maximum_speed - abs(speed)) / (maximum_speed - base_speed),
        )
        return math.copysign(
            min(abs(torque), DESIGN.wheel_torque_limit * authority), torque
        )

    def run(self, experiment, render=False, output=None):
        self.reset(experiment.initial(self.model))
        if experiment.obstacle:
            obstacles = experiment.obstacle
            if isinstance(obstacles[0], (int, float)):
                obstacles = (obstacles,)
            if len(obstacles) > 3:
                raise ValueError("the MuJoCo plant provides three terrain features")
            for index, obstacle in enumerate(obstacles, start=1):
                center, width, height = obstacle[:3]
                heading = obstacle[3] if len(obstacle) >= 4 else 0.0
                self.place_obstacle(
                    center * math.cos(heading),
                    height,
                    width / 2,
                    index,
                    y=center * math.sin(heading),
                    heading=heading,
                )
        experiment.controller.reset()

        viewer = None
        if render:
            from mujoco import viewer as mj_viewer
            viewer = mj_viewer.launch_passive(self.model, self.data)

        rows = []
        next_sample = 0.0
        while self.data.time < experiment.duration:
            state = State(self)
            role_control, target, phase = experiment.control(state)
            physical_control = self.roles.to_physical(role_control)
            self.apply(physical_control)

            self.data.xfrc_applied[:] = 0.0
            experiment.disturb(self.model, self.data, state)
            mujoco.mj_step(self.model, self.data)

            if self.data.time >= next_sample:
                rows.append(
                    state.row(target, phase, role_control, physical_control)
                )
                next_sample += 0.02
            if viewer is not None:
                viewer.sync()
                time.sleep(self.model.opt.timestep)

        if viewer is not None:
            viewer.close()
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
        return rows
