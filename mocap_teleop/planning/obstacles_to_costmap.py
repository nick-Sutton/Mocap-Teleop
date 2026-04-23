"""
obstacles_to_costmap.py — Convert /obstacles MarkerArray to a nav_msgs/OccupancyGrid.

Sim-mode bridge: the CBF safety filter already receives circular obstacle
primitives via /obstacles.  This node turns those same primitives into a
nav_msgs/OccupancyGrid on /local_costmap so the MPPI planner can query it
without a real depth camera.

On hardware this node is replaced by the RealSense → depthimage_to_laserscan
→ costmap pipeline; /local_costmap is published by that pipeline instead.

Grid is fixed in world frame — large enough to cover the operating area.
MPPI queries it with world-frame coordinates, so no robot-pose tracking needed.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData
from visualization_msgs.msg import MarkerArray
from builtin_interfaces.msg import Time


class ObstaclesToCostmap(Node):
    def __init__(self):
        super().__init__('obstacles_to_costmap')

        self.declare_parameter('resolution',  0.1)    # m per cell
        self.declare_parameter('grid_size',   20.0)   # half-width of grid (m)
        self.declare_parameter('origin_x',    -10.0)  # world X of cell (0,0)
        self.declare_parameter('origin_y',    -10.0)  # world Y of cell (0,0)
        self.declare_parameter('clearance',   0.4)    # obstacle inflation radius (m)
        self.declare_parameter('lethal_cost', 100)    # cell value inside obstacles
        self.declare_parameter('publish_hz',  10.0)   # republish rate

        self._res       = self.get_parameter('resolution').value
        self._size      = self.get_parameter('grid_size').value
        self._origin_x  = self.get_parameter('origin_x').value
        self._origin_y  = self.get_parameter('origin_y').value
        self._clearance = self.get_parameter('clearance').value
        self._lethal    = int(self.get_parameter('lethal_cost').value)
        hz              = self.get_parameter('publish_hz').value

        self._width  = int(2 * self._size / self._res)
        self._height = int(2 * self._size / self._res)

        # Current obstacle list — replaced atomically each callback
        self._obstacles: list = []   # list of (cx, cy, radius)

        self.create_subscription(
            MarkerArray, '/obstacles',
            self._obstacles_cb, 10)

        self._pub = self.create_publisher(OccupancyGrid, '/local_costmap', 1)
        self.create_timer(1.0 / hz, self._publish_cb)

        self.get_logger().info(
            f'obstacles_to_costmap ready  '
            f'grid={self._width}×{self._height} cells  '
            f'res={self._res} m  clearance={self._clearance} m')

    def _obstacles_cb(self, msg: MarkerArray) -> None:
        obs = []
        for marker in msg.markers:
            cx = marker.pose.position.x
            cy = marker.pose.position.y
            r  = marker.scale.x / 2.0
            if r > 0.01:
                obs.append((cx, cy, r))
        self._obstacles = obs

    def _publish_cb(self) -> None:
        grid = np.zeros(self._width * self._height, dtype=np.int8)

        for cx, cy, radius in self._obstacles:
            inflated_r = radius + self._clearance
            # Mark all cells whose centres fall within the inflated radius
            r_cells = int(np.ceil(inflated_r / self._res)) + 1
            col_c = int((cx - self._origin_x) / self._res)
            row_c = int((cy - self._origin_y) / self._res)

            for dc in range(-r_cells, r_cells + 1):
                for dr in range(-r_cells, r_cells + 1):
                    col = col_c + dc
                    row = row_c + dr
                    if not (0 <= col < self._width and 0 <= row < self._height):
                        continue
                    wx = self._origin_x + (col + 0.5) * self._res
                    wy = self._origin_y + (row + 0.5) * self._res
                    dist = np.hypot(wx - cx, wy - cy)
                    if dist <= inflated_r:
                        grid[row * self._width + col] = self._lethal

        msg              = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'

        info             = MapMetaData()
        info.resolution  = self._res
        info.width       = self._width
        info.height      = self._height
        info.origin.position.x = self._origin_x
        info.origin.position.y = self._origin_y
        msg.info         = info
        msg.data         = grid.tolist()

        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObstaclesToCostmap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
