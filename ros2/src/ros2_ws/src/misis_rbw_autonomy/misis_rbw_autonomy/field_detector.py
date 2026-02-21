from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]
Rect = Tuple[int, int, int, int]
BgrColor = Tuple[int, int, int]


@dataclass
class TrackedObject:
    last_positions: Deque[Point] = field(default_factory=lambda: deque(maxlen=10))
    last_seen: float = 0.0
    confidence: float = 0.0
    track_id: int = 0
    object_type: str = "unknown"


@dataclass(frozen=True)
class DetectedObject:
    object_type: str
    center_m: Point
    center_px: Point
    rect: Rect
    size_m: Point
    area_m2: float
    confidence: float = 1.0
    track_id: Optional[int] = None
    color: Optional[str] = None
    display_color: BgrColor = (255, 255, 255)


@dataclass(frozen=True)
class FieldObservation:
    toys: List[DetectedObject]
    robots: List[DetectedObject]
    debug_frame: np.ndarray
    homography_ready: bool
    background_ready: bool


@dataclass
class DetectionConfig:
    min_area_px: int = 300
    min_fill_ratio: float = 0.5
    max_aspect_ratio: float = 5.0
    toy_min_area_m2: float = 0.0016
    toy_max_area_m2: float = 0.0036
    robot_min_area_m2: float = 0.04
    robot_max_area_m2: float = 0.09
    robot_min_confidence: float = 0.3
    toy_min_confidence: float = 0.3
    robot_max_speed_mps: float = 3.0
    track_max_age_s: float = 0.5
    calibration_frames: int = 30
    field_width_m: float = 1.75
    field_height_m: float = 1.25
    marker_coords_m: Dict[int, Point] = field(
        default_factory=lambda: {
            0: (0.0325, 0.9675),
            1: (0.0325, 0.4675),
            15: (1.5325, 0.9675),
            16: (1.5325, 0.4675),
            4: (0.2825, 0.2175),
            2: (0.2825, 1.2175),
            12: (1.2825, 1.2175),
            14: (1.2825, 0.2175),
        }
    )


class FieldDetector:
    """Detects ArUco field coordinates, colored game objects, and moving robots."""

    def __init__(self, config: Optional[DetectionConfig] = None) -> None:
        self.config = config or DetectionConfig()
        self.field_corners = np.array(
            [
                [0.0, 0.0],
                [self.config.field_width_m, 0.0],
                [self.config.field_width_m, self.config.field_height_m],
                [0.0, self.config.field_height_m],
            ],
            dtype=np.float32,
        )

        self.homography: Optional[np.ndarray] = None
        self.frame_count = 0
        self.object_tracks: Dict[int, TrackedObject] = {}
        self.next_id = 0
        self.background_calibrated = False
        self.calibration_counter = 0

        self.color_ranges = {
            "red": [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([170, 100, 100]), np.array([180, 255, 255])),
            ],
            "green": [(np.array([40, 40, 40]), np.array([80, 255, 255]))],
            "blue": [(np.array([100, 100, 100]), np.array([140, 255, 255]))],
            "yellow": [(np.array([20, 100, 100]), np.array([35, 255, 255]))],
        }
        self.display_colors: Dict[str, BgrColor] = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
        }

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=20,
            detectShadows=True,
        )
        self.aruco_detector = self._create_aruco_detector()

    def _create_aruco_detector(self):
        if not hasattr(cv2, "aruco"):
            return None
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        parameters.adaptiveThreshConstant = 7
        if hasattr(cv2.aruco, "ArucoDetector"):
            return cv2.aruco.ArucoDetector(dictionary, parameters)
        return dictionary, parameters

    def reset_background(self) -> None:
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=20,
            detectShadows=True,
        )
        self.background_calibrated = False
        self.calibration_counter = 0
        self.object_tracks.clear()

    def detect_aruco(self, frame: np.ndarray):
        if self.aruco_detector is None:
            return [], None
        if isinstance(self.aruco_detector, tuple):
            dictionary, parameters = self.aruco_detector
            corners, ids, _ = cv2.aruco.detectMarkers(
                frame,
                dictionary,
                parameters=parameters,
            )
            return corners, ids
        corners, ids, _ = self.aruco_detector.detectMarkers(frame)
        return corners, ids

    @staticmethod
    def get_left_bottom_corner(corners: np.ndarray) -> np.ndarray:
        min_x = np.min(corners[:, 0])
        candidates = [pt for pt in corners if pt[0] == min_x]
        return max(candidates, key=lambda p: p[1])

    def compute_homography(self, corners, ids) -> Optional[np.ndarray]:
        if ids is None or len(ids) < 4:
            return None

        pixel_points = []
        object_points = []
        for idx, marker_id in enumerate(ids.flatten()):
            if marker_id not in self.config.marker_coords_m:
                continue
            corner_points = corners[idx][0]
            pixel_points.append(self.get_left_bottom_corner(corner_points))
            object_points.append(self.config.marker_coords_m[int(marker_id)])

        if len(pixel_points) < 4:
            return None

        homography, _ = cv2.findHomography(
            np.array(pixel_points, dtype=np.float32),
            np.array(object_points, dtype=np.float32),
            cv2.RANSAC,
            5.0,
        )
        return homography

    def pixel_to_meters(self, point_px: Point) -> Optional[Point]:
        if self.homography is None:
            return None
        point = np.array([[point_px]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.homography)
        x, y = transformed[0][0]
        return float(x), float(y)

    def object_size_meters(self, rect: Rect) -> Tuple[float, float, float]:
        if self.homography is None:
            return 0.0, 0.0, 0.0

        x, y, w, h = rect
        corners_px = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        corners_m = cv2.perspectiveTransform(corners_px, self.homography).reshape(-1, 2)
        width_m = float(np.linalg.norm(corners_m[1] - corners_m[0]))
        height_m = float(np.linalg.norm(corners_m[2] - corners_m[1]))
        return width_m, height_m, width_m * height_m

    def is_inside_field(self, point_m: Point) -> bool:
        x, y = point_m
        margin = 0.03
        return (
            -margin <= x <= self.config.field_width_m + margin
            and -margin <= y <= self.config.field_height_m + margin
        )

    def detect_toys(self, frame: np.ndarray) -> List[DetectedObject]:
        if self.homography is None:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        toys: List[DetectedObject] = []

        for color_name, ranges in self.color_ranges.items():
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area_px = cv2.contourArea(contour)
                if area_px < self.config.min_area_px:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect = max(w, h) / (min(w, h) + 1e-6)
                if aspect > self.config.max_aspect_ratio:
                    continue

                fill_ratio = area_px / (w * h + 1e-6)
                if fill_ratio < self.config.min_fill_ratio:
                    continue

                center_px = (x + w / 2.0, y + h / 2.0)
                center_m = self.pixel_to_meters(center_px)
                if center_m is None or not self.is_inside_field(center_m):
                    continue

                width_m, height_m, area_m2 = self.object_size_meters((x, y, w, h))
                if self.config.toy_min_area_m2 <= area_m2 <= self.config.toy_max_area_m2:
                    toys.append(
                        DetectedObject(
                            object_type="toy",
                            center_m=center_m,
                            center_px=center_px,
                            rect=(x, y, w, h),
                            size_m=(width_m, height_m),
                            area_m2=area_m2,
                            color=color_name,
                            display_color=self.display_colors[color_name],
                        )
                    )

        return toys

    def detect_robots(self, frame: np.ndarray, toys: Iterable[DetectedObject]) -> List[DetectedObject]:
        if self.homography is None or not self.background_calibrated:
            return []

        fg_mask = self.bg_subtractor.apply(frame)
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        _, fg_mask = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)

        known_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for toy in toys:
            x, y, w, h = toy.rect
            margin = 15
            x1, y1 = max(0, x - margin), max(0, y - margin)
            x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
            cv2.rectangle(known_mask, (x1, y1), (x2, y2), 255, -1)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        robots: List[DetectedObject] = []
        for contour in contours:
            area_px = cv2.contourArea(contour)
            if area_px < self.config.min_area_px:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            roi = known_mask[y : y + h, x : x + w]
            if roi.size and np.mean(roi) > 20:
                continue

            center_px = (x + w / 2.0, y + h / 2.0)
            center_m = self.pixel_to_meters(center_px)
            if center_m is None or not self.is_inside_field(center_m):
                continue

            width_m, height_m, area_m2 = self.object_size_meters((x, y, w, h))
            if not (self.config.robot_min_area_m2 <= area_m2 <= self.config.robot_max_area_m2):
                continue

            aspect_m = max(width_m, height_m) / (min(width_m, height_m) + 1e-6)
            if aspect_m <= 1.5:
                robots.append(
                    DetectedObject(
                        object_type="robot",
                        center_m=center_m,
                        center_px=center_px,
                        rect=(x, y, w, h),
                        size_m=(width_m, height_m),
                        area_m2=area_m2,
                        display_color=(0, 165, 255),
                    )
                )

        return robots

    def update_tracking(
        self,
        objects: Iterable[DetectedObject],
        current_time: float,
    ) -> List[DetectedObject]:
        confirmed: List[DetectedObject] = []
        active_tracks: Dict[int, TrackedObject] = {}

        for obj in objects:
            best_match: Optional[int] = None
            best_dist = float("inf")
            for track_id, track in self.object_tracks.items():
                if track.object_type != obj.object_type or not track.last_positions:
                    continue
                last_position = track.last_positions[-1]
                dist = float(np.hypot(obj.center_m[0] - last_position[0], obj.center_m[1] - last_position[1]))
                dt = current_time - track.last_seen
                if dt > 0 and dist / dt > self.config.robot_max_speed_mps:
                    continue
                if dist < best_dist and dist < 0.5:
                    best_dist = dist
                    best_match = track_id

            if best_match is None:
                best_match = self.next_id
                self.next_id += 1
                track = TrackedObject(
                    last_seen=current_time,
                    confidence=0.2,
                    track_id=best_match,
                    object_type=obj.object_type,
                )
                self.object_tracks[best_match] = track
            else:
                track = self.object_tracks[best_match]
                track.confidence = min(1.0, track.confidence + 0.15)
                track.last_seen = current_time

            track.last_positions.append(obj.center_m)
            active_tracks[best_match] = track

            min_confidence = (
                self.config.robot_min_confidence
                if obj.object_type == "robot"
                else self.config.toy_min_confidence
            )
            tracked = DetectedObject(
                object_type=obj.object_type,
                center_m=obj.center_m,
                center_px=obj.center_px,
                rect=obj.rect,
                size_m=obj.size_m,
                area_m2=obj.area_m2,
                confidence=track.confidence,
                track_id=best_match,
                color=obj.color,
                display_color=obj.display_color,
            )
            if track.confidence >= min_confidence:
                confirmed.append(tracked)

        for track_id, track in list(self.object_tracks.items()):
            if track_id in active_tracks:
                continue
            if current_time - track.last_seen > self.config.track_max_age_s:
                del self.object_tracks[track_id]
            else:
                track.confidence = max(0.0, track.confidence - 0.05)

        return confirmed

    def draw_debug(
        self,
        frame: np.ndarray,
        toys: Iterable[DetectedObject],
        robots: Iterable[DetectedObject],
    ) -> np.ndarray:
        debug = frame.copy()

        if self.homography is not None:
            inv_h = np.linalg.inv(self.homography)
            corners_img = cv2.perspectiveTransform(self.field_corners.reshape(-1, 1, 2), inv_h)
            corners_img = corners_img.reshape(-1, 2).astype(int)
            cv2.polylines(debug, [corners_img], True, (0, 255, 0), 2)

        for toy in toys:
            x, y, w, h = toy.rect
            color = toy.display_color
            cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)
            center_px = int(toy.center_px[0]), int(toy.center_px[1])
            cv2.drawMarker(debug, center_px, color, cv2.MARKER_CROSS, 10, 2)
            label = (
                f"{toy.color or 'toy'} #{toy.track_id} "
                f"({toy.center_m[0]:.2f},{toy.center_m[1]:.2f})"
            )
            cv2.putText(debug, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        for robot in robots:
            x, y, w, h = robot.rect
            color = (0, 0, 255) if robot.confidence > 0.6 else (0, 165, 255)
            cv2.rectangle(debug, (x, y), (x + w, y + h), color, 3)
            center_px = int(robot.center_px[0]), int(robot.center_px[1])
            cv2.circle(debug, center_px, 8, color, -1)
            label = (
                f"robot #{robot.track_id} "
                f"({robot.center_m[0]:.2f},{robot.center_m[1]:.2f}) "
                f"{robot.confidence:.0%}"
            )
            cv2.putText(debug, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        info = [
            f"Frame: {self.frame_count}",
            f"Toys: {len(list(toys)) if not isinstance(toys, list) else len(toys)}",
            f"Robots: {len(list(robots)) if not isinstance(robots, list) else len(robots)}",
            f"Tracks: {len(self.object_tracks)}",
            f"Homography: {'YES' if self.homography is not None else 'NO'}",
            (
                "BG: DONE"
                if self.background_calibrated
                else f"BG: {self.calibration_counter}/{self.config.calibration_frames}"
            ),
        ]
        for idx, text in enumerate(info):
            cv2.putText(debug, text, (10, 30 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return debug

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> FieldObservation:
        self.frame_count += 1
        now = timestamp if timestamp is not None else time.monotonic()

        corners, ids = self.detect_aruco(frame)
        homography = self.compute_homography(corners, ids)
        if homography is not None:
            self.homography = homography

        if not self.background_calibrated:
            self.bg_subtractor.apply(frame)
            self.calibration_counter += 1
            if self.calibration_counter >= self.config.calibration_frames:
                self.background_calibrated = True
            debug = self.draw_debug(frame, [], [])
            cv2.putText(
                debug,
                f"CALIBRATING {self.calibration_counter}/{self.config.calibration_frames}",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            return FieldObservation(
                toys=[],
                robots=[],
                debug_frame=debug,
                homography_ready=self.homography is not None,
                background_ready=False,
            )

        toys = self.detect_toys(frame)
        robots = self.detect_robots(frame, toys)
        confirmed = self.update_tracking([*toys, *robots], now)
        confirmed_toys = [obj for obj in confirmed if obj.object_type == "toy"]
        confirmed_robots = [obj for obj in confirmed if obj.object_type == "robot"]
        debug = self.draw_debug(frame, confirmed_toys, confirmed_robots)

        return FieldObservation(
            toys=confirmed_toys,
            robots=confirmed_robots,
            debug_frame=debug,
            homography_ready=self.homography is not None,
            background_ready=True,
        )
