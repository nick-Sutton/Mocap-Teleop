#!/usr/bin/env python3

import os
import sys
from scipy.spatial.transform import Rotation

import mocap_teleop.util.io_parser as io

# Load controller paths from config and add them to sys.path so that
# high_level_cmd (and any modules it depends on) can be found regardless
# of the working directory at launch time.
_ctrl_cfg = io.parse_controller_config()
sys.path.append(os.path.expanduser(_ctrl_cfg['controller_path']))
sys.path.append(os.path.expanduser(_ctrl_cfg['interface_subpath']))

import high_level_cmd as cmd


class CtrlInterface():
    def __init__(self):
        pass

    @staticmethod
    def walk(vx=0, vy=0, vrz=0):
        """walk command"""
        cmd.walk_idqp(h=0.25, vx=vx, vy=vy, vrz=vrz)

    @staticmethod
    def stand(rx=0, ry=0, rz=0):
        """Stand command"""
        cmd.stand_idqp(h=0.25, rx=rx, ry=ry, rz=rz)

    @staticmethod
    def soft_stop():
        """Soft stop"""
        cmd.soft_stop()

    @staticmethod
    def hard_stop():
        """Hard stop"""
        cmd.hard_stop()

    @staticmethod
    def get_tlm_data():
        """Get telemetry data"""
        return cmd.get_tlm_data()

    @staticmethod
    def get_robot_orientation():
        """Get the robot's current orientation as quaternion"""
        tlm_data = cmd.get_tlm_data()
        if isinstance(tlm_data, list):
            r_roll, r_pitch, r_yaw = tlm_data[0]["q"][3:6]
        else:
            r_roll, r_pitch, r_yaw = tlm_data["q"][3:6]
        return Rotation.from_euler("xyz", [r_roll, r_pitch, r_yaw]).as_quat()

    @staticmethod
    def get_robot_position():
        """Get the robot's current position (x, y, z)"""
        tlm_data = cmd.get_tlm_data()
        if isinstance(tlm_data, list):
            position = tlm_data[0]["q"][:3]
        else:
            position = tlm_data["q"][:3]
        return position

    @staticmethod
    def is_robot_state_unsafe():
        """Check if robot state is unsafe"""
        return False  # Placeholder
