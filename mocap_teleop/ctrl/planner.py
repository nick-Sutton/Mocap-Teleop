"""
planner.py — Lightweight 2-D A* planner for the PLAN FSM state.

Builds a coarse occupancy grid from the current obstacle list and returns
the next waypoint toward the goal.  Replanning is cheap enough to run at
a few Hz as the human (goal) moves.

Usage:
    planner = AStarPlanner(grid_res=0.2, grid_size=20.0)
    waypoint = planner.next_waypoint(robot_pos, goal_pos, obstacles)
    # Returns None if no path found (goal unreachable).
"""

import heapq
import numpy as np
from mocap_teleop.ctrl.cbf_qp import Obstacle

# Lookahead distance along planned path returned as next waypoint.
_WAYPOINT_LOOKAHEAD = 0.5   # m


class AStarPlanner:
    def __init__(self, grid_res: float = 0.2, grid_size: float = 20.0,
                 clearance: float = 0.4):
        """
        grid_res  : cell size in metres
        grid_size : half-width of grid (grid spans ±grid_size from origin)
        clearance : obstacle inflation radius in metres
        """
        self.res       = grid_res
        self.size      = grid_size
        self.clearance = clearance
        self._n        = int(2 * grid_size / grid_res)

    # ── Public API ────────────────────────────────────────────────────────────

    def next_waypoint(
        self,
        robot_pos:  np.ndarray,   # [x, y] world frame
        goal_pos:   np.ndarray,   # [x, y] world frame (human current position)
        obstacles:  list,         # list of Obstacle
    ) -> np.ndarray | None:
        """
        Plan from robot_pos to goal_pos avoiding obstacles.
        Returns the next waypoint _WAYPOINT_LOOKAHEAD metres along the path,
        or goal_pos itself if the path is shorter.  Returns None if no path.
        """
        grid   = self._build_grid(robot_pos, obstacles)
        path   = self._astar(grid, robot_pos, goal_pos)
        if path is None:
            return None
        return self._lookahead_waypoint(path)

    # ── Grid construction ─────────────────────────────────────────────────────

    def _build_grid(self, origin: np.ndarray, obstacles: list) -> np.ndarray:
        """Binary occupancy grid centred on the robot's current position."""
        grid = np.zeros((self._n, self._n), dtype=bool)
        for obs in obstacles:
            inflated_r = obs.radius + self.clearance
            # Grid cells whose centres are within inflated radius are occupied
            cx, cy = self._world_to_cell(obs.center, origin)
            r_cells = int(np.ceil(inflated_r / self.res))
            for dx in range(-r_cells, r_cells + 1):
                for dy in range(-r_cells, r_cells + 1):
                    if dx*dx + dy*dy <= r_cells*r_cells:
                        ix, iy = cx + dx, cy + dy
                        if 0 <= ix < self._n and 0 <= iy < self._n:
                            grid[ix, iy] = True
        return grid

    # ── A* ────────────────────────────────────────────────────────────────────

    def _astar(
        self, grid: np.ndarray,
        start: np.ndarray, goal: np.ndarray,
    ) -> list | None:
        origin  = start
        si      = self._world_to_cell(start, origin)
        gi      = self._world_to_cell(goal,  origin)

        # Clamp goal to grid
        gi = (np.clip(gi[0], 0, self._n - 1),
              np.clip(gi[1], 0, self._n - 1))

        # If goal cell is occupied, find nearest free cell
        if grid[gi]:
            gi = self._nearest_free(grid, gi)
            if gi is None:
                return None

        open_heap = []
        heapq.heappush(open_heap, (0.0, si))
        came_from = {si: None}
        g_cost    = {si: 0.0}

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == gi:
                return self._reconstruct(came_from, cur, origin)

            for nb in self._neighbours(cur):
                if grid[nb]:
                    continue
                step  = self.res * (1.414 if nb[0] != cur[0] and nb[1] != cur[1] else 1.0)
                ng    = g_cost[cur] + step
                if ng < g_cost.get(nb, float('inf')):
                    g_cost[nb]  = ng
                    h           = self.res * np.hypot(gi[0] - nb[0], gi[1] - nb[1])
                    heapq.heappush(open_heap, (ng + h, nb))
                    came_from[nb] = cur

        return None   # no path found

    def _neighbours(self, cell):
        x, y = cell
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self._n and 0 <= ny < self._n:
                    yield (nx, ny)

    def _reconstruct(self, came_from, current, origin):
        path = []
        while current is not None:
            path.append(self._cell_to_world(current, origin))
            current = came_from[current]
        path.reverse()
        return path

    def _nearest_free(self, grid, cell):
        for r in range(1, self._n // 2):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        ix, iy = cell[0] + dx, cell[1] + dy
                        if 0 <= ix < self._n and 0 <= iy < self._n:
                            if not grid[ix, iy]:
                                return (ix, iy)
        return None

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _world_to_cell(self, pos: np.ndarray, origin: np.ndarray):
        offset = pos - origin + self.size
        ix = int(np.clip(offset[0] / self.res, 0, self._n - 1))
        iy = int(np.clip(offset[1] / self.res, 0, self._n - 1))
        return (ix, iy)

    def _cell_to_world(self, cell, origin: np.ndarray) -> np.ndarray:
        x = cell[0] * self.res - self.size + origin[0]
        y = cell[1] * self.res - self.size + origin[1]
        return np.array([x, y])

    # ── Waypoint extraction ───────────────────────────────────────────────────

    def _lookahead_waypoint(self, path: list) -> np.ndarray:
        if len(path) == 1:
            return path[0]
        dist = 0.0
        for i in range(1, len(path)):
            dist += np.linalg.norm(path[i] - path[i - 1])
            if dist >= _WAYPOINT_LOOKAHEAD:
                return path[i]
        return path[-1]
