from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from misis_rbw_autonomy.field_detector import DetectedObject


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


@dataclass(frozen=True)
class RobotCommand:
    linear: float
    angular: float
    target: Optional[str]
    reason: str


class CompetitionStrategy:
    """Reactive 90-second strategy for collecting colored objects on the field."""

    def __init__(
        self,
        max_linear: float = 0.24,
        max_angular: float = 1.4,
        match_duration_s: float = 90.0,
        return_home_s: float = 12.0,
        home_pose: Pose2D = Pose2D(0.18, 0.18, 0.0),
        pickup_radius_m: float = 0.13,
        obstacle_radius_m: float = 0.30,
    ) -> None:
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.match_duration_s = match_duration_s
        self.return_home_s = return_home_s
        self.home_pose = home_pose
        self.pickup_radius_m = pickup_radius_m
        self.obstacle_radius_m = obstacle_radius_m
        self.color_scores = {
            "red": 3.0,
            "yellow": 2.5,
            "blue": 2.0,
            "green": 1.5,
        }
        self.start_time: Optional[float] = None
        self.last_pose: Optional[Pose2D] = None

    def reset_match(self) -> None:
        self.start_time = None
        self.last_pose = None

    def compute_command(
        self,
        toys: Iterable[DetectedObject],
        robots: Iterable[DetectedObject],
        now_s: float,
        own_pose: Optional[Pose2D] = None,
    ) -> RobotCommand:
        if self.start_time is None:
            self.start_time = now_s
        elapsed = now_s - self.start_time
        if elapsed >= self.match_duration_s:
            return RobotCommand(0.0, 0.0, None, "match time is over")

        robots_list = list(robots)
        pose = own_pose or self.estimate_own_pose(robots_list)
        if pose is None:
            return RobotCommand(0.0, 0.45 * self.max_angular, None, "searching own robot")

        if elapsed >= self.match_duration_s - self.return_home_s:
            return self.drive_to_pose(pose, self.home_pose, "returning to start zone")

        toys_list = list(toys)
        target = self.choose_target(pose, toys_list)
        if target is None:
            return RobotCommand(0.0, 0.35 * self.max_angular, None, "searching objects")

        target_point = self.apply_obstacle_avoidance(pose, target.center_m, robots_list)
        command = self.drive_to_point(
            pose,
            target_point,
            f"{target.color or 'object'} #{target.track_id}",
        )
        if self.distance((pose.x, pose.y), target.center_m) < self.pickup_radius_m:
            return RobotCommand(0.08, 0.0, command.target, "pickup/score approach")
        return command

    def estimate_own_pose(self, robots: List[DetectedObject]) -> Optional[Pose2D]:
        if not robots:
            return self.last_pose

        if self.last_pose is None:
            own = min(
                robots,
                key=lambda obj: self.distance(obj.center_m, (self.home_pose.x, self.home_pose.y)),
            )
            pose = Pose2D(own.center_m[0], own.center_m[1], self.home_pose.yaw)
            self.last_pose = pose
            return pose

        own = min(
            robots,
            key=lambda obj: self.distance(obj.center_m, (self.last_pose.x, self.last_pose.y)),
        )
        dx = own.center_m[0] - self.last_pose.x
        dy = own.center_m[1] - self.last_pose.y
        yaw = self.last_pose.yaw
        if math.hypot(dx, dy) > 0.02:
            yaw = math.atan2(dy, dx)
        pose = Pose2D(own.center_m[0], own.center_m[1], yaw)
        self.last_pose = pose
        return pose

    def choose_target(self, pose: Pose2D, toys: List[DetectedObject]) -> Optional[DetectedObject]:
        if not toys:
            return None

        def value(toy: DetectedObject) -> float:
            distance = self.distance((pose.x, pose.y), toy.center_m)
            score = self.color_scores.get(toy.color or "", 1.0)
            confidence = max(0.2, toy.confidence)
            return score * confidence / (distance + 0.18)

        return max(toys, key=value)

    def apply_obstacle_avoidance(
        self,
        pose: Pose2D,
        target: Tuple[float, float],
        robots: Iterable[DetectedObject],
    ) -> Tuple[float, float]:
        vx = target[0] - pose.x
        vy = target[1] - pose.y
        for robot in robots:
            dist = self.distance((pose.x, pose.y), robot.center_m)
            if dist < 1e-6 or dist > self.obstacle_radius_m:
                continue
            weight = (self.obstacle_radius_m - dist) / self.obstacle_radius_m
            vx += weight * (pose.x - robot.center_m[0]) * 0.8
            vy += weight * (pose.y - robot.center_m[1]) * 0.8
        return pose.x + vx, pose.y + vy

    def drive_to_pose(self, pose: Pose2D, target: Pose2D, reason: str) -> RobotCommand:
        return self.drive_to_point(pose, (target.x, target.y), "home", reason)

    def drive_to_point(
        self,
        pose: Pose2D,
        target: Tuple[float, float],
        target_name: str,
        reason: str = "driving to target",
    ) -> RobotCommand:
        dx = target[0] - pose.x
        dy = target[1] - pose.y
        distance = math.hypot(dx, dy)
        if distance < 0.05:
            return RobotCommand(0.0, 0.0, target_name, f"{reason}: reached")

        target_yaw = math.atan2(dy, dx)
        yaw_error = self.wrap_angle(target_yaw - pose.yaw)
        angular = self.clamp(2.2 * yaw_error, -self.max_angular, self.max_angular)
        heading_gate = max(0.0, 1.0 - abs(yaw_error) / 1.2)
        linear = self.clamp(distance * 0.75, 0.08, self.max_linear) * heading_gate
        return RobotCommand(linear, angular, target_name, reason)

    @staticmethod
    def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def wrap_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))
