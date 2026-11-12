"""
RehabVision - Pose Detection Module
Supports both Upper-Limb (Arm/Elbow) and Lower-Limb (Leg/Knee) tracking.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseDetector:
    def __init__(self, model_path="pose_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # Standard MediaPipe body skeleton connections
        self.connections = [
            (11, 12), (11, 23), (12, 24), (23, 24), # Torso
            (11, 13), (13, 15),                     # Left Arm
            (12, 14), (14, 16),                     # Right Arm
            (23, 25), (25, 27),                     # Left Leg
            (24, 26), (26, 28)                      # Right Leg
        ]

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        if detection_result.pose_landmarks:
            return detection_result.pose_landmarks[0]
        return None

    def get_best_arm_landmarks(self, landmarks, frame_width, frame_height):
        """
        Extracts Shoulder, Elbow, and Wrist for the best visible arm.
        Left Arm: (11, 13, 15) | Right Arm: (12, 14, 16)
        """
        if not landmarks:
            return None

        # Compare visibility of Left Arm vs Right Arm
        left_score = landmarks[11].visibility + landmarks[13].visibility + landmarks[15].visibility
        right_score = landmarks[12].visibility + landmarks[14].visibility + landmarks[16].visibility

        side = "Left" if left_score >= right_score else "Right"
        sh_id, el_id, wr_id = (11, 13, 15) if side == "Left" else (12, 14, 16)

        lm_sh = landmarks[sh_id]
        lm_el = landmarks[el_id]
        lm_wr = landmarks[wr_id]

        return {
            "side": side,
            "shoulder": (int(lm_sh.x * frame_width), int(lm_sh.y * frame_height)),
            "elbow": (int(lm_el.x * frame_width), int(lm_el.y * frame_height)),
            "wrist": (int(lm_wr.x * frame_width), int(lm_wr.y * frame_height)),
        }

    def get_best_knee_landmarks(self, landmarks, frame_width, frame_height):
        """
        Extracts Hip, Knee, and Ankle for the best visible leg.
        Left Leg: (23, 25, 27) | Right Leg: (24, 26, 28)
        """
        if not landmarks:
            return None

        left_score = landmarks[23].visibility + landmarks[25].visibility + landmarks[27].visibility
        right_score = landmarks[24].visibility + landmarks[26].visibility + landmarks[28].visibility

        side = "Left" if left_score >= right_score else "Right"
        hip_id, knee_id, ankle_id = (23, 25, 27) if side == "Left" else (24, 26, 28)

        lm_hip = landmarks[hip_id]
        lm_knee = landmarks[knee_id]
        lm_ankle = landmarks[ankle_id]

        return {
            "side": side,
            "hip": (int(lm_hip.x * frame_width), int(lm_hip.y * frame_height)),
            "knee": (int(lm_knee.x * frame_width), int(lm_knee.y * frame_height)),
            "ankle": (int(lm_ankle.x * frame_width), int(lm_ankle.y * frame_height)),
        }