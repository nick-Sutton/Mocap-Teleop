"""
feasibility_estimator.py — Wrap the DDPG critic as a deployment-time
feasibility signal.

Q(s, u_safe) estimates the expected cumulative imitation reward from the
current state.  Low Q → the critic predicts that the robot cannot track the
human from here → trigger the BLOCKED → PLAN transition.

The state vector layout must match build_state() in train_amcbf.py exactly:
    [d_goal_body(2), vel_body(2), obs0(4), obs1(4), obs2(4)]
"""

import numpy as np
import torch
import torch.nn as nn

from mocap_teleop.ctrl.cbf_qp import Obstacle, cbf_value

# Must match train_amcbf.py constants
_STATE_DIM = 4 + 3 * 4   # 3 obstacles * 4 features
_D_SCALE   = 5.0
_V_MAX     = 1.5
_R_SCALE   = 1.0
_H_SCALE   = 2.0


class _CriticNet(nn.Module):
    # Architecture must match CriticNet in train_amcbf.py exactly —
    # action is injected at the second layer, not concatenated at the input.
    def __init__(self, state_dim: int = _STATE_DIM, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden + 2, 64)   # +2 for action [vx, vy]
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(torch.cat([x, action], dim=-1)))
        return self.fc3(x)


class FeasibilityEstimator:
    """Load the trained critic and evaluate Q at each control step."""

    def __init__(self, critic_path: str, device: str = 'cpu'):
        self._device = torch.device(device)
        self._net = _CriticNet().to(self._device)
        ckpt = torch.load(critic_path, map_location=self._device)
        # Checkpoint may be a full training dict or just the state_dict
        state_dict = ckpt.get('critic', ckpt)
        self._net.load_state_dict(state_dict)
        self._net.eval()

    def q_value(
        self,
        robot_pos:  np.ndarray,   # [x, y]
        robot_yaw:  float,
        vel_body:   np.ndarray,   # [vx, vy]
        goal_xy:    np.ndarray,   # human current position [x, y]
        obstacles:  list,         # list of Obstacle
        u_safe:     np.ndarray,   # [vx, vy] — CBF-filtered action
    ) -> float:
        state = _build_state(robot_pos, robot_yaw, vel_body, goal_xy, obstacles)
        s = torch.tensor(state,  dtype=torch.float32, device=self._device).unsqueeze(0)
        a = torch.tensor(u_safe[:2].astype(np.float32),
                         device=self._device).unsqueeze(0)
        with torch.no_grad():
            return float(self._net(s, a).item())


def _build_state(
    pos_xy:    np.ndarray,
    yaw:       float,
    vel_body:  np.ndarray,
    goal_xy:   np.ndarray,
    obstacles: list,
) -> np.ndarray:
    d_goal_world = goal_xy - pos_xy
    c, s         = np.cos(yaw), np.sin(yaw)
    d_goal_body  = np.array([
        c * d_goal_world[0] + s * d_goal_world[1],
        -s * d_goal_world[0] + c * d_goal_world[1],
    ])

    feats = np.zeros(_STATE_DIM, dtype=np.float32)
    feats[0:2] = d_goal_body  / _D_SCALE
    feats[2:4] = vel_body[:2] / _V_MAX

    for i in range(3):
        base = 4 + i * 4
        if i < len(obstacles):
            obs = obstacles[i]
            d_world = obs.center - pos_xy
            d_body  = np.array([
                c * d_world[0] + s * d_world[1],
                -s * d_world[0] + c * d_world[1],
            ])
            h = cbf_value(pos_xy, obs.center, yaw, obs.radius)
            feats[base + 0] = d_body[0]          / _D_SCALE
            feats[base + 1] = d_body[1]          / _D_SCALE
            feats[base + 2] = obs.radius         / _R_SCALE
            feats[base + 3] = np.clip(h, -1.0, _H_SCALE) / _H_SCALE
        else:
            feats[base + 3] = 1.0   # padded slot: very safe

    return feats
