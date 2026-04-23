"""
mppi_planner.py — Model Predictive Path Integral planner for FSM PLAN/RESYNC states.

Replaces the A* + WaypointFollower pair.  Given a local occupancy grid (from the
RealSense depth camera), the robot's current state, and the human's position as
the goal, MPPI samples N trajectory rollouts under a unicycle kinematic model,
weights them by exp(-cost/lambda), and returns the velocity command for the
first step of the optimal trajectory.

Unicycle model (body frame, dt per step):
    x'   = x + vx * cos(yaw) * dt  -  vy * sin(yaw) * dt
    y'   = y + vx * sin(yaw) * dt  +  vy * cos(yaw) * dt
    yaw' = yaw + vrz * dt

Cost terms (all per-step, summed over horizon):
    goal_cost      : distance from rollout endpoint to human position
    obstacle_cost  : costmap value at each rollout cell (high inside obstacles)
    smoothness_cost: L2 norm of per-step control (penalises jerky commands)
    terminal_cost  : extra weight on distance at the final step

The warm-start shifts the previous optimal control sequence one step forward
each tick so the planner is consistent between calls.
"""

import numpy as np


# ── Defaults ──────────────────────────────────────────────────────────────────
_N_SAMPLES    = 256     # number of sampled trajectories
_HORIZON      = 20      # steps
_DT           = 0.1     # seconds per step  (planner runs at ~10 Hz)
_LAMBDA       = 1.0     # temperature: lower → greedier weighting
_SIGMA_VX     = 0.5     # exploration noise std for vx (m/s)
_SIGMA_VY     = 0.2     # exploration noise std for vy (m/s)
_SIGMA_VRZ    = 0.4     # exploration noise std for vrz (rad/s)

_W_GOAL       = 2.0     # weight: terminal distance to goal
_W_OBSTACLE   = 5.0     # weight: costmap cost (0–100 scale)
_W_SMOOTH     = 0.05    # weight: control effort

_VX_MAX       = 1.5     # m/s
_VY_MAX       = 0.4     # m/s  — Go2 lateral limit
_VRZ_MAX      = 1.5     # rad/s

# Costmap values above this threshold are treated as fully occupied.
_OBS_THRESH   = 50      # out of 100


class OccupancyGrid:
    """Minimal wrapper around a nav_msgs/OccupancyGrid snapshot.

    Stores the data as a numpy array and exposes a cost() method for
    world-frame queries.  The caller (teleop_node) instantiates this from
    the ROS message and passes it to MPPIPlanner.compute().
    """

    def __init__(
        self,
        data:       np.ndarray,   # 1-D int8 array, row-major (height × width)
        width:      int,
        height:     int,
        resolution: float,        # metres per cell
        origin_x:   float,        # world X of cell (0, 0)
        origin_y:   float,        # world Y of cell (0, 0)
    ):
        self._grid      = data.reshape((height, width)).astype(np.float32)
        self._width     = width
        self._height    = height
        self._res       = resolution
        self._origin_x  = origin_x
        self._origin_y  = origin_y

    def cost(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return costmap values for world-frame arrays x, y (same shape).

        Returns 100.0 for out-of-bounds or unknown (-1) cells.
        """
        col = ((x - self._origin_x) / self._res).astype(int)
        row = ((y - self._origin_y) / self._res).astype(int)

        in_bounds = (
            (col >= 0) & (col < self._width) &
            (row >= 0) & (row < self._height)
        )

        result = np.full(x.shape, 100.0, dtype=np.float32)
        ib = in_bounds
        vals = self._grid[row[ib], col[ib]]
        # nav_msgs/OccupancyGrid uses -1 for unknown — treat as obstacle
        vals = np.where(vals < 0, 100.0, vals)
        result[ib] = vals
        return result


class MPPIPlanner:
    """MPPI planner: samples trajectories and returns the first-step cmd_vel."""

    def __init__(
        self,
        n_samples:    int   = _N_SAMPLES,
        horizon:      int   = _HORIZON,
        dt:           float = _DT,
        lam:          float = _LAMBDA,
        sigma_vx:     float = _SIGMA_VX,
        sigma_vy:     float = _SIGMA_VY,
        sigma_vrz:    float = _SIGMA_VRZ,
        w_goal:       float = _W_GOAL,
        w_obstacle:   float = _W_OBSTACLE,
        w_smooth:     float = _W_SMOOTH,
        vx_max:       float = _VX_MAX,
        vy_max:       float = _VY_MAX,
        vrz_max:      float = _VRZ_MAX,
    ):
        self._N       = n_samples
        self._H       = horizon
        self._dt      = dt
        self._lam     = lam
        self._sigmas  = np.array([sigma_vx, sigma_vy, sigma_vrz])
        self._w_goal  = w_goal
        self._w_obs   = w_obstacle
        self._w_smooth = w_smooth
        self._limits  = np.array([vx_max, vy_max, vrz_max])

        # Warm-start: previous optimal control sequence [H, 3]
        self._u_nom = np.zeros((horizon, 3))

    def compute(
        self,
        robot_pos:  np.ndarray,        # [x, y]
        robot_yaw:  float,
        goal_xy:    np.ndarray,        # [x, y]  human position
        grid:       OccupancyGrid | None,
    ) -> np.ndarray:                   # [vx_body, vy_body, vrz]
        """Run one MPPI iteration and return the command for this step.

        If grid is None (no costmap available yet) the obstacle term is skipped.
        """
        N, H, dt = self._N, self._H, self._dt

        # ── Sample perturbations [N, H, 3] ───────────────────────────────────
        eps = np.random.randn(N, H, 3) * self._sigmas[None, None, :]

        # Perturbed controls = warm-start + noise, clipped to limits
        u = np.clip(
            self._u_nom[None, :, :] + eps,   # [N, H, 3]
            -self._limits, self._limits,
        )

        # ── Rollout unicycle kinematics [N, H, 3] ────────────────────────────
        px  = np.full(N, robot_pos[0])
        py  = np.full(N, robot_pos[1])
        yaw = np.full(N, robot_yaw)

        costs = np.zeros(N)

        for h in range(H):
            vx_b  = u[:, h, 0]
            vy_b  = u[:, h, 1]
            vrz   = u[:, h, 2]

            c, s  = np.cos(yaw), np.sin(yaw)
            px   += (c * vx_b - s * vy_b) * dt
            py   += (s * vx_b + c * vy_b) * dt
            yaw  += vrz * dt

            # Obstacle cost at each step
            if grid is not None:
                obs_val = grid.cost(px, py)       # [N]
                costs += self._w_obs * (obs_val / 100.0)

            # Per-step control smoothness
            costs += self._w_smooth * np.sum(u[:, h, :] ** 2, axis=1)

        # Terminal goal cost
        costs += self._w_goal * np.hypot(px - goal_xy[0], py - goal_xy[1])

        # ── MPPI weight update ────────────────────────────────────────────────
        beta    = costs.min()
        weights = np.exp(-(costs - beta) / self._lam)
        weights /= weights.sum() + 1e-8

        # Weighted average of perturbations → update nominal sequence
        delta_u = np.einsum('n,nhc->hc', weights, eps)   # [H, 3]
        self._u_nom = np.clip(
            self._u_nom + delta_u,
            -self._limits, self._limits,
        )

        # Extract first command, then shift warm-start for next call
        cmd = self._u_nom[0].copy()
        self._u_nom = np.roll(self._u_nom, -1, axis=0)
        self._u_nom[-1] = 0.0   # zero-pad last step

        return cmd   # [vx_body, vy_body, vrz]

    def reset(self) -> None:
        """Clear warm-start — call when re-entering PLAN from IMITATE."""
        self._u_nom[:] = 0.0
