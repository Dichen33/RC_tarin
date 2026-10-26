from __future__ import annotations

import math

import numpy as np


class NavigationController:
    """Waypoint navigation adapted from the reference sim2sim controller."""

    def __init__(self, arrival_radius: float = 0.15):
        self.arrival_radius = arrival_radius
        self.mode = "MANUAL"
        self.mission_name = "None"
        self.waypoints: list[tuple[float, float, float, str]] = []
        self.waypoint_index = 0
        self.target_x: float | None = None
        self.target_y: float | None = None
        self.drawn_path: list[tuple[float, float]] = []

    @property
    def active(self) -> bool:
        return self.mode == "AUTO" and self.target_x is not None and self.target_y is not None

    def start_mission(self, name: str, waypoints: list[tuple[float, float, float, str]]) -> None:
        self.waypoints = list(waypoints)
        self.waypoint_index = 0
        self.mission_name = name
        self.mode = "AUTO" if self.waypoints else "MANUAL"
        self.drawn_path = [(wp[0], wp[1]) for wp in self.waypoints]
        if self.waypoints:
            self.target_x, self.target_y = self.waypoints[0][0], self.waypoints[0][1]
        else:
            self.target_x = self.target_y = None

    def start_target(self, x: float, y: float, speed: float = 0.25) -> None:
        self.start_mission("Mouse target", [(float(x), float(y), speed, "rough")])

    def cancel(self) -> None:
        self.mode = "MANUAL"
        self.mission_name = "None"
        self.waypoints.clear()
        self.waypoint_index = 0
        self.target_x = self.target_y = None
        self.drawn_path.clear()

    def update(self, robot_x: float, robot_y: float, robot_yaw: float) -> np.ndarray:
        """Return a body-frame command [vx, vy, yaw] for the current waypoint."""
        if not self.active:
            return np.zeros(3, dtype=np.float32)

        dx = self.target_x - robot_x
        dy = self.target_y - robot_y
        distance = math.hypot(dx, dy)

        if distance < self.arrival_radius:
            if self.waypoint_index + 1 < len(self.waypoints):
                self.waypoint_index += 1
                next_waypoint = self.waypoints[self.waypoint_index]
                self.target_x, self.target_y = next_waypoint[0], next_waypoint[1]
                return np.zeros(3, dtype=np.float32)
            self.cancel()
            return np.zeros(3, dtype=np.float32)

        waypoint = self.waypoints[self.waypoint_index]
        speed_limit = float(np.clip(waypoint[2], 0.05, 1.2))
        target_yaw = math.atan2(dy, dx)
        yaw_error = (target_yaw - robot_yaw + math.pi) % (2.0 * math.pi) - math.pi
        yaw_command = float(np.clip(1.8 * yaw_error, -1.0, 1.0))

        # Turn in place when the target is more than roughly 45 degrees away.
        if abs(yaw_error) > 0.8:
            forward_command = 0.0
        else:
            forward_command = float(np.clip(0.8 * distance * math.cos(yaw_error), -0.2, speed_limit))
        return np.asarray((forward_command, 0.0, yaw_command), dtype=np.float32)
