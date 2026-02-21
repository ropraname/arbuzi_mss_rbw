#!/usr/bin/env python3
"""
Простой скрипт для детекции игрушек и роботов на шахматном поле.
Игрушки: цветные объекты размером 4-6 см
Роботы: движущиеся объекты размером 20-30 см
"""

import cv2
import numpy as np
import json
import time
from collections import deque
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

@dataclass
class TrackedObject:
    """Класс для отслеживания объектов"""
    last_positions: deque = field(default_factory=lambda: deque(maxlen=10))
    last_seen: float = 0
    confidence: float = 0
    id: int = 0
    object_type: str = "unknown"

class SimpleDetector:
    def __init__(self):
        # Параметры детекции
        self.min_area = 300  # минимальная площадь в пикселях
        self.min_fill_ratio = 0.5
        self.max_aspect_ratio = 5.0
        
        # Размеры объектов в метрах
        self.toy_min_area = 0.0016  # 4x4 см
        self.toy_max_area = 0.0036  # 6x6 см
        self.robot_min_area = 0.04   # 20x20 см
        self.robot_max_area = 0.09   # 30x30 см
        
        # Параметры трекинга
        self.robot_min_confidence = 0.3
        self.toy_min_confidence = 0.3
        self.robot_max_speed = 3.0  # м/с
        
        # Цветовые диапазоны для игрушек (HSV)
        self.color_ranges = {
            'red': {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255]), 'color': (0, 0, 255)},
            'green': {'lower': np.array([40, 40, 40]), 'upper': np.array([80, 255, 255]), 'color': (0, 255, 0)},
            'blue': {'lower': np.array([100, 100, 100]), 'upper': np.array([140, 255, 255]), 'color': (255, 0, 0)},
            'yellow': {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255]), 'color': (0, 255, 255)},
        }
        
        # Координаты ArUco маркеров (левый нижний угол) в метрах
        self.marker_coords = {
            0: (0.0325, 0.9675),
            1: (0.0325, 0.4675),
            15: (1.5325, 0.9675),
            16: (1.5325, 0.4675),
            4: (0.2825, 0.2175),
            2: (0.2825, 1.2175),
            12: (1.2825, 1.2175),
            14: (1.2825, 0.2175)
        }
        
        # Границы поля (метры)
        self.field_corners = np.array([
            [0.0, 0.0], [1.75, 0.0], [1.75, 1.25], [0.0, 1.25]
        ], dtype=np.float32)
        
        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.adaptiveThreshConstant = 7
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        # Фоновый вычитатель для детекции движения
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=20, detectShadows=True
        )
        
        # Состояние
        self.homography = None
        self.frame_count = 0
        self.object_tracks = {}
        self.next_id = 0
        self.background_calibrated = False
        self.calibration_counter = 0
        self.calibration_frames = 30
        
        print("✅ Детектор инициализирован")
        print(f"🎯 Игрушки: {self.toy_min_area*10000:.0f}-{self.toy_max_area*10000:.0f} см²")
        print(f"🎯 Роботы: {self.robot_min_area*10000:.0f}-{self.robot_max_area*10000:.0f} см²")
        print("🔄 Начинается калибровка фона (30 кадров)...")
    
    def detect_aruco(self, frame):
        """Детекция ArUco маркеров"""
        corners, ids, _ = self.detector.detectMarkers(frame)
        return corners, ids
    
    def get_left_bottom_corner(self, corners):
        """Получение левого нижнего угла маркера"""
        pts = corners
        min_x_idx = np.argmin(pts[:, 0])
        min_x = pts[min_x_idx, 0]
        candidates = [pt for pt in pts if pt[0] == min_x]
        return max(candidates, key=lambda p: p[1])
    
    def compute_homography(self, corners, ids):
        """Вычисление гомографии по маркерам"""
        if ids is None or len(ids) < 4:
            return None
        
        pixel_points = []
        object_points = []
        
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in self.marker_coords:
                corner_pts = corners[i][0]
                left_bottom = self.get_left_bottom_corner(corner_pts)
                pixel_points.append(left_bottom)
                object_points.append(self.marker_coords[marker_id])
        
        if len(pixel_points) < 4:
            return None
        
        pixel_points = np.array(pixel_points, dtype=np.float32)
        object_points = np.array(object_points, dtype=np.float32)
        
        H, _ = cv2.findHomography(pixel_points, object_points, cv2.RANSAC, 5.0)
        return H
    
    def pixel_to_meters(self, px):
        """Перевод пикселей в метры"""
        if self.homography is None:
            return None
        pt = np.array([[px[0], px[1]]], dtype=np.float32).reshape(-1, 1, 2)
        pt_m = cv2.perspectiveTransform(pt, self.homography)
        return (pt_m[0][0][0], pt_m[0][0][1])
    
    def get_object_size_meters(self, rect, H):
        """Получение размера объекта в метрах"""
        x, y, w, h = rect
        corners_px = np.array([
            [x, y], [x + w, y], [x + w, y + h], [x, y + h]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        corners_m = cv2.perspectiveTransform(corners_px, H)
        width_m = abs(corners_m[1][0][0] - corners_m[0][0][0])
        height_m = abs(corners_m[2][0][1] - corners_m[0][0][1])
        return width_m, height_m, width_m * height_m
    
    def is_inside_field(self, point_m):
        """Проверка, находится ли точка внутри поля"""
        x, y = point_m
        corners = self.field_corners
        inside = False
        for i in range(len(corners)):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % len(corners)]
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                inside = not inside
        return inside
    
    def detect_toys(self, frame):
        """Детекция игрушек (цветные объекты 4-6 см)"""
        if self.homography is None:
            return []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        toys = []
        
        for name, ranges in self.color_ranges.items():
            mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area_px = cv2.contourArea(cnt)
                if area_px < self.min_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = max(w, h) / (min(w, h) + 1e-6)
                if aspect > self.max_aspect_ratio:
                    continue
                
                fill_ratio = area_px / (w * h + 1e-6)
                if fill_ratio < self.min_fill_ratio:
                    continue
                
                center_px = (x + w/2, y + h/2)
                center_m = self.pixel_to_meters(center_px)
                
                if center_m and self.is_inside_field(center_m):
                    width_m, height_m, area_m = self.get_object_size_meters((x, y, w, h), self.homography)
                    
                    # Проверяем размер игрушки (4-6 см)
                    if self.toy_min_area <= area_m <= self.toy_max_area:
                        toys.append({
                            'rect': (x, y, w, h),
                            'center_px': center_px,
                            'center_m': center_m,
                            'size_m': (width_m, height_m),
                            'area_m': area_m,
                            'color': name,
                            'display_color': ranges['color'],
                            'object_type': 'toy'
                        })
        
        return toys
    
    def detect_robots(self, frame, toys):
        """Детекция роботов (движущиеся объекты 20-30 см)"""
        if self.homography is None or not self.background_calibrated:
            return []
        
        # Маска движения
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Морфологическая обработка
        kernel = np.ones((5,5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        _, fg_mask = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)
        
        # Маска известных объектов (игрушки)
        known_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for toy in toys:
            x, y, w, h = toy['rect']
            margin = 15
            x1, y1 = max(0, x - margin), max(0, y - margin)
            x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
            cv2.rectangle(known_mask, (x1, y1), (x2, y2), 255, -1)
        
        # Находим контуры движения
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        robots = []
        
        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < self.min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Проверяем пересечение с известными объектами
            roi = known_mask[y:y+h, x:x+w]
            if np.mean(roi) > 20:
                continue
            
            center_px = (x + w/2, y + h/2)
            center_m = self.pixel_to_meters(center_px)
            
            if center_m and self.is_inside_field(center_m):
                width_m, height_m, area_m = self.get_object_size_meters((x, y, w, h), self.homography)
                
                # Проверяем размер робота (20-30 см)
                if self.robot_min_area <= area_m <= self.robot_max_area:
                    aspect = max(width_m, height_m) / (min(width_m, height_m) + 1e-6)
                    if aspect <= 1.5:  # Примерно квадратный
                        robots.append({
                            'rect': (x, y, w, h),
                            'center_px': center_px,
                            'center_m': center_m,
                            'size_m': (width_m, height_m),
                            'area_m': area_m,
                            'object_type': 'robot'
                        })
        
        return robots
    
    def update_tracking(self, objects, current_time):
        """Обновление трекинга объектов"""
        confirmed = []
        active_tracks = {}
        
        for obj in objects:
            center = obj['center_m']
            obj_type = obj['object_type']
            best_match = None
            best_dist = float('inf')
            
            # Поиск ближайшего трека
            for tid, track in self.object_tracks.items():
                if track.object_type != obj_type:
                    continue
                if track.last_positions:
                    last_pos = track.last_positions[-1]
                    dist = np.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
                    
                    if len(track.last_positions) > 1:
                        dt = current_time - track.last_seen
                        if dt > 0:
                            speed = dist / dt
                            if speed > self.robot_max_speed:
                                continue
                    
                    if dist < best_dist and dist < 0.5:
                        best_dist = dist
                        best_match = tid
            
            if best_match is not None:
                track = self.object_tracks[best_match]
                track.last_positions.append(center)
                track.last_seen = current_time
                track.confidence = min(1.0, track.confidence + 0.15)
                active_tracks[best_match] = track
                obj['track_id'] = best_match
                obj['confidence'] = track.confidence
            else:
                track = TrackedObject()
                track.last_positions.append(center)
                track.last_seen = current_time
                track.confidence = 0.2
                track.id = self.next_id
                track.object_type = obj_type
                self.object_tracks[self.next_id] = track
                obj['track_id'] = self.next_id
                obj['confidence'] = track.confidence
                self.next_id += 1
                active_tracks[self.next_id - 1] = track
        
        # Удаление старых треков
        to_delete = []
        for tid, track in self.object_tracks.items():
            if tid not in active_tracks:
                if current_time - track.last_seen > 0.5:
                    to_delete.append(tid)
                else:
                    track.confidence = max(0, track.confidence - 0.05)
                    active_tracks[tid] = track
        
        for tid in to_delete:
            del self.object_tracks[tid]
        
        # Фильтрация по уверенности
        for obj in objects:
            tid = obj.get('track_id')
            if tid and tid in active_tracks:
                track = active_tracks[tid]
                if obj['object_type'] == 'robot' and track.confidence >= self.robot_min_confidence:
                    confirmed.append(obj)
                elif obj['object_type'] == 'toy' and track.confidence >= self.toy_min_confidence:
                    confirmed.append(obj)
        
        return confirmed
    
    def draw_debug(self, frame, toys, robots):
        """Рисование отладочной информации"""
        debug = frame.copy()
        
        # Рисуем границы поля
        if self.homography is not None:
            H_inv = np.linalg.inv(self.homography)
            corners_img = cv2.perspectiveTransform(self.field_corners.reshape(-1, 1, 2), H_inv)
            corners_img = corners_img.reshape(-1, 2).astype(int)
            cv2.polylines(debug, [corners_img], True, (0, 255, 0), 2)
        
        # Рисуем игрушки
        for toy in toys:
            x, y, w, h = toy['rect']
            color = toy['display_color']
            cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)
            cx, cy = int(toy['center_px'][0]), int(toy['center_px'][1])
            cv2.drawMarker(debug, (cx, cy), color, cv2.MARKER_CROSS, 10, 2)
            
            label = f"{toy['color']} ({toy['center_m'][0]:.2f},{toy['center_m'][1]:.2f})"
            label += f" {toy['size_m'][0]*100:.0f}x{toy['size_m'][1]*100:.0f}cm"
            cv2.putText(debug, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Рисуем роботов
        for robot in robots:
            x, y, w, h = robot['rect']
            conf = robot.get('confidence', 0)
            if conf > 0.6:
                color = (0, 0, 255)
            elif conf > 0.3:
                color = (0, 165, 255)
            else:
                color = (0, 255, 255)
            
            cv2.rectangle(debug, (x, y), (x + w, y + h), color, 3)
            cx, cy = int(robot['center_px'][0]), int(robot['center_px'][1])
            cv2.circle(debug, (cx, cy), 8, color, -1)
            
            label = f"ROBOT #{robot.get('track_id', '?')} ({robot['center_m'][0]:.2f},{robot['center_m'][1]:.2f})"
            label += f" {robot['size_m'][0]*100:.0f}x{robot['size_m'][1]*100:.0f}cm [{conf:.0%}]"
            cv2.putText(debug, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Информационная панель
        info = [
            f"Frame: {self.frame_count}",
            f"Toys: {len(toys)}",
            f"Robots: {len(robots)}",
            f"Tracks: {len(self.object_tracks)}",
            f"Homography: {'YES' if self.homography is not None else 'NO'}",
            f"BG Calib: {'DONE' if self.background_calibrated else f'{self.calibration_counter}/{self.calibration_frames}'}"
        ]
        
        for i, text in enumerate(info):
            cv2.putText(debug, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug
    
    def process_frame(self, frame):
        """Обработка одного кадра"""
        self.frame_count += 1
        
        # Калибровка фона
        if not self.background_calibrated:
            self.bg_subtractor.apply(frame)
            self.calibration_counter += 1
            if self.calibration_counter >= self.calibration_frames:
                self.background_calibrated = True
                print(f"✅ Калибровка фона завершена!")
            
            debug = frame.copy()
            cv2.putText(debug, f"CALIBRATING... {self.calibration_counter}/{self.calibration_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return debug, [], []
        
        # Детекция ArUco и гомографии
        corners, ids = self.detect_aruco(frame)
        if ids is not None and len(ids) >= 4:
            H = self.compute_homography(corners, ids)
            if H is not None:
                self.homography = H
        
        # Детекция объектов
        toys = self.detect_toys(frame)
        robots = self.detect_robots(frame, toys)
        
        # Трекинг
        current_time = time.time()
        all_objects = toys + robots
        confirmed = self.update_tracking(all_objects, current_time)
        
        # Разделяем
        confirmed_toys = [obj for obj in confirmed if obj['object_type'] == 'toy']
        confirmed_robots = [obj for obj in confirmed if obj['object_type'] == 'robot']
        
        # Визуализация
        debug = self.draw_debug(frame, confirmed_toys, confirmed_robots)
        
        return debug, confirmed_toys, confirmed_robots
    
    def run(self, source=0):
        """Запуск детектора"""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"❌ Не удалось открыть источник видео: {source}")
            return
        
        print("🎥 Видео открыто")
        print("🟢 Управление: 'q' - выход, 's' - скриншот, 'r' - сброс калибровки")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            debug, toys, robots = self.process_frame(frame)
            cv2.imshow('Detector', debug)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"capture_{timestamp}.jpg", debug)
                print(f"📸 Скриншот сохранен")
            elif key == ord('r'):
                print("🔄 Сброс калибровки...")
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=20, detectShadows=True)
                self.background_calibrated = False
                self.calibration_counter = 0
                self.object_tracks.clear()
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Завершено")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Детектор игрушек и роботов')
    parser.add_argument('--source', '-s', type=str, default='0',
                        help='Источник видео (0 - камера, или путь к файлу)')
    args = parser.parse_args()
    
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    detector = SimpleDetector()
    detector.run(source)

if __name__ == '__main__':
    main()