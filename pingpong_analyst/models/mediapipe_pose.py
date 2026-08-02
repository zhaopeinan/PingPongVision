"""
MediaPipe Pose Landmarker 封装
- 用于计算肩、肘、腕的精确3D角度
- 评估发力机制 (撞击vs摩擦)
"""
from typing import Any

import numpy as np
from loguru import logger

from .base import BaseModelWrapper
from ..utils.device_manager import DeviceInfo, DeviceType


class MediaPipePoseAnalyzer(BaseModelWrapper):
    """MediaPipe Pose Landmarker (Heavy) 封装"""

    # MediaPipe 33个关键点索引
    KEYPOINT_INDICES = {
        "nose": 0,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24,
    }

    def load(self) -> bool:
        if self._loaded:
            return True

        if not self.config.get("enabled", True):
            logger.info("MediaPipe 已禁用")
            return False

        try:
            import mediapipe as mp
            self._mp = mp
            self._mp_pose = mp.solutions.pose

            model_complexity = 2  # Heavy model
            if self.device_info.device_type in (DeviceType.CPU, DeviceType.MPS):
                model_complexity = 1  # MediaPipe solutions API 仍主要使用 CPU
                logger.info("MediaPipe 使用中等复杂度 (CPU/MPS模式)")

            self._model = self._mp_pose.Pose(
                # 每个人框都是独立裁剪图，必须独立检测，不能让上一位选手
                # 的 tracking 状态影响下一位选手。
                static_image_mode=self.config.get("static_image_mode", True),
                model_complexity=model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=self.config.get("min_detection_confidence", 0.5),
                min_tracking_confidence=self.config.get("min_tracking_confidence", 0.5),
            )
            self._loaded = True
            logger.info("MediaPipe Pose Landmarker 加载成功")
            return True
        except Exception as e:
            logger.error(f"MediaPipe 加载失败: {e}")
            return False

    def infer(self, frame: np.ndarray) -> Any:
        """单帧推理, 返回 MediaPipe landmarks"""
        if not self._loaded:
            raise RuntimeError("模型未加载")

        import cv2
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._model.process(rgb_frame)
        return results

    def extract_landmarks(self, frame: np.ndarray) -> dict | None:
        """
        提取3D关键点

        Returns:
            {
                "landmarks": np.ndarray[33, 4],  # [x, y, z, visibility]
                "world_landmarks": np.ndarray[33, 4],  # 真实世界坐标(厘米)
            }
        """
        results = self.infer(frame)
        if results.pose_landmarks is None:
            return None

        landmarks = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
        )
        world_landmarks = None
        if results.pose_world_landmarks is not None:
            world_landmarks = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_world_landmarks.landmark]
            )

        return {
            "landmarks": landmarks,
            "world_landmarks": world_landmarks,
        }

    def enrich_persons(self, frame: np.ndarray, persons: list[dict]) -> list[dict]:
        """对每个 YOLO 人框单独提取姿态，并把结果写回该 person。"""
        if not persons or not self.is_loaded:
            return persons

        import cv2

        height, width = frame.shape[:2]
        enriched = []
        margin_ratio = float(self.config.get("crop_margin_ratio", 0.08))

        for person in persons:
            item = dict(person)
            bbox = person.get("bbox")
            if bbox is None or len(bbox) != 4:
                enriched.append(item)
                continue

            x1, y1, x2, y2 = [float(value) for value in bbox]
            box_width = max(x2 - x1, 1.0)
            box_height = max(y2 - y1, 1.0)
            x1 = max(0, int(x1 - box_width * margin_ratio))
            y1 = max(0, int(y1 - box_height * margin_ratio))
            x2 = min(width, int(x2 + box_width * margin_ratio))
            y2 = min(height, int(y2 + box_height * margin_ratio))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                enriched.append(item)
                continue

            result = self.extract_landmarks(crop)
            if result is None:
                enriched.append(item)
                continue

            landmarks = result["landmarks"].copy()
            crop_width = max(x2 - x1, 1)
            crop_height = max(y2 - y1, 1)
            landmarks[:, 0] = (x1 + landmarks[:, 0] * crop_width) / width
            landmarks[:, 1] = (y1 + landmarks[:, 1] * crop_height) / height
            item["pose_landmarks"] = landmarks
            item["arm_angles"] = self.analyze_arm_angles(landmarks)
            item["body_metrics"] = self.analyze_body_metrics(landmarks)
            enriched.append(item)

        return enriched

    @staticmethod
    def select_display_angles(persons: list[dict]) -> dict | None:
        """选择置信度最高的选手角度，维持旧版单字典前端协议。"""
        candidates = [
            person for person in persons
            if person.get("arm_angles")
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda person: person.get("conf", 0.0)).get("arm_angles")

    @staticmethod
    def calculate_joint_angle(p1, p2, p3) -> float:
        """
        计算3点形成的关节角度

        Args:
            p1, p2, p3: 3D坐标点, p2为关节顶点
        Returns:
            角度 (度)
        """
        p1 = np.array(p1[:3])
        p2 = np.array(p2[:3])
        p3 = np.array(p3[:3])

        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def analyze_arm_angles(self, landmarks: np.ndarray) -> dict:
        """
        分析手臂关节角度 (评估发力机制)

        Returns:
            {
                "left_elbow_angle": float,
                "right_elbow_angle": float,
                "left_shoulder_angle": float,
                "right_shoulder_angle": float,
            }
        """
        idx = self.KEYPOINT_INDICES

        result = {}
        # 肘关节: 肩-肘-腕
        if landmarks is not None:
            result["left_elbow_angle"] = self.calculate_joint_angle(
                landmarks[idx["left_shoulder"]],
                landmarks[idx["left_elbow"]],
                landmarks[idx["left_wrist"]],
            )
            result["right_elbow_angle"] = self.calculate_joint_angle(
                landmarks[idx["right_shoulder"]],
                landmarks[idx["right_elbow"]],
                landmarks[idx["right_wrist"]],
            )
            # 肩关节: 髋-肩-肘
            result["left_shoulder_angle"] = self.calculate_joint_angle(
                landmarks[idx["left_hip"]],
                landmarks[idx["left_shoulder"]],
                landmarks[idx["left_elbow"]],
            )
            result["right_shoulder_angle"] = self.calculate_joint_angle(
                landmarks[idx["right_hip"]],
                landmarks[idx["right_shoulder"]],
                landmarks[idx["right_elbow"]],
            )

        return result

    def analyze_body_metrics(self, landmarks: np.ndarray) -> dict:
        """计算动作分析专用的躯干指标。

        这些指标只在 ``action`` 入口调用，回合剪辑不会加载或执行姿态模型。
        ``torso_lean_angle`` 以竖直方向为 0 度，向右倾斜为正值。
        """
        if landmarks is None or len(landmarks) <= max(self.KEYPOINT_INDICES.values()):
            return {}

        idx = self.KEYPOINT_INDICES
        left_shoulder = landmarks[idx["left_shoulder"], :2]
        right_shoulder = landmarks[idx["right_shoulder"], :2]
        left_hip = landmarks[idx["left_hip"], :2]
        right_hip = landmarks[idx["right_hip"], :2]

        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        hip_center = (left_hip + right_hip) / 2.0
        torso = shoulder_center - hip_center
        torso_length = float(np.linalg.norm(torso))
        if torso_length < 1e-6:
            return {}

        lean_angle = float(np.degrees(np.arctan2(torso[0], -torso[1])))
        return {
            "torso_lean_angle": lean_angle,
            "torso_length": torso_length,
            "shoulder_width": float(np.linalg.norm(left_shoulder - right_shoulder)),
            "hip_width": float(np.linalg.norm(left_hip - right_hip)),
        }
