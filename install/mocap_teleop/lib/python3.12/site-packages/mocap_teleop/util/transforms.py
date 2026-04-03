"""
ct_math.py — Math utilities for offline CSV replay.

In the online ROS2 path these functions are NOT used — velocities are computed
inside mocap_driver_node and arrive via TwistStamped messages.

These functions are retained for run_offline_mode in teleop.py so that the
offline CSV replay path continues to work without ROS2.
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def body_velocities(
    curr_pos:  np.ndarray,
    prev_pos:  np.ndarray,
    curr_quat: np.ndarray,
    prev_quat: np.ndarray,
    dt:        float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute linear and angular body velocities by finite difference.

    Parameters
    ----------
    curr_pos / prev_pos   : [x, y, z]
    curr_quat / prev_quat : [qx, qy, qz, qw]
    dt                    : time step in seconds

    Returns
    -------
    linear_vel  : [vx, vy, vz]  in world frame
    angular_vel : [wx, wy, wz]  in world frame (rotation vector / dt)
    """
    linear_vel  = (curr_pos - prev_pos) / dt

    rot_curr    = Rotation.from_quat(curr_quat)
    rot_prev    = Rotation.from_quat(prev_quat)
    angular_vel = (rot_curr * rot_prev.inv()).as_rotvec() / dt

    return linear_vel, angular_vel


def optitrack_pos_to_ros(x: float, y: float, z: float) -> tuple:
    """
    Transform a position from OptiTrack Y-up to ROS Z-up convention.
    Used only when loading raw CSV files from Motive.
    """
    return z, -x, y


def optitrack_quat_to_ros(
    qx: float, qy: float, qz: float, qw: float
) -> tuple:
    """
    Transform a quaternion to match the OptiTrack → ROS position axis swap.
    """
    return qz, -qx, qy, qw


def apply_coordinate_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply OptiTrack → ROS coordinate transform to all rigid bodies in a
    Motive-exported CSV DataFrame.

    Column naming convention: RigidBodyName:Position:X, etc.

    Used only in offline preprocessing / dataset preparation.
    """
    df_out = df.copy()

    rigid_body_names: set[str] = set()
    for col in df.columns:
        parts = col.split(':')
        if len(parts) >= 2:
            rigid_body_names.add(parts[0])

    print(f"Transforming {len(rigid_body_names)} rigid bodies...")

    for rb in rigid_body_names:
        px, py, pz = f"{rb}:Position:X", f"{rb}:Position:Y", f"{rb}:Position:Z"
        if all(c in df.columns for c in [px, py, pz]):
            ox, oy, oz = df[px].values.copy(), df[py].values.copy(), df[pz].values.copy()
            df_out[px] =  oz    # new X = old Z
            df_out[py] = -ox   # new Y = -old X
            df_out[pz] =  oy   # new Z = old Y

        rx = f"{rb}:Rotation:X"; ry = f"{rb}:Rotation:Y"
        rz = f"{rb}:Rotation:Z"; rw = f"{rb}:Rotation:W"
        if all(c in df.columns for c in [rx, ry, rz, rw]):
            oqx = df[rx].values.copy(); oqy = df[ry].values.copy()
            oqz = df[rz].values.copy(); oqw = df[rw].values.copy()
            df_out[rx] =  oqz   # new qx = old qz
            df_out[ry] = -oqx  # new qy = -old qx
            df_out[rz] =  oqy  # new qz = old qy
            df_out[rw] =  oqw  # qw unchanged

    return df_out

def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw angle from a quaternion."""
    return np.arctan2(2.0 * (qw * qz + qx * qy),
                        1.0 - 2.0 * (qy**2 + qz**2))

def normalize_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return np.arctan2(np.sin(angle), np.cos(angle))

def world_to_body(world_vel: np.ndarray, heading: float) -> np.ndarray:
    """Rotate 2-D velocity from world frame into robot body frame."""
    c, s = np.cos(heading), np.sin(heading)
    return np.array([ c * world_vel[0] + s * world_vel[1],
                        -s * world_vel[0] + c * world_vel[1]])