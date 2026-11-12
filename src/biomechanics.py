"""
RehabVision - Biomechanics Engine
Calculates joint angles, classifies movement states, filters noise, and tracks repetitions.
"""

import math

def calculate_joint_angle(point_a, point_b, point_c):
    """
    Calculates the 2D angle (in degrees) between three joint coordinates.
    point_a: (x, y) - Proximal joint (e.g., Hip or Shoulder)
    point_b: (x, y) - Center vertex joint (e.g., Knee or Elbow)
    point_c: (x, y) - Distal joint (e.g., Ankle or Wrist)
    """
    radians_a = math.atan2(point_a[1] - point_b[1], point_a[0] - point_b[0])
    radians_c = math.atan2(point_c[1] - point_b[1], point_c[0] - point_b[0])
    
    angle = abs(math.degrees(radians_a - radians_c))
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return round(angle, 1)


def classify_knee_posture(angle):
    """
    Evaluates knee flexion angle for lower-limb rehabilitation.
    """
    if angle >= 165.0:
        return "Standing Straight / Full Extension"
    elif 135.0 <= angle < 165.0:
        return "Quarter Squat (Gentle Rehab)"
    elif 90.0 <= angle < 135.0:
        return "Half Squat (Functional Strength)"
    else:
        return "Deep Squat (High Flexion)"


def classify_elbow_posture(angle):
    """
    Evaluates arm/elbow flexion angle for upper-limb rehabilitation.
    """
    if angle >= 155.0:
        return "Arm Straight / Full Extension"
    elif 75.0 <= angle < 155.0:
        return "Half Flexion (90 deg Bend)"
    else:
        return "Full Flexion (Bicep Curl)"


class AngleSmoother:
    """
    Biomedical filter that removes frame-by-frame sensor noise
    using an Exponential Moving Average (EMA).
    """
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.smoothed_angle = None

    def update(self, raw_angle):
        if self.smoothed_angle is None:
            self.smoothed_angle = raw_angle
        else:
            self.smoothed_angle = (self.alpha * raw_angle) + ((1.0 - self.alpha) * self.smoothed_angle)
        return round(self.smoothed_angle, 1)


class ExerciseTracker:
    """
    Biomedical state machine that counts complete exercise repetitions
    and tracks the patient's deepest Range of Motion (ROM).
    """
    def __init__(self, mode="ARM"):
        self.mode = mode
        self.counter = 0
        self.stage = "EXTENDED"
        self.best_flexion = 180.0

    def set_mode(self, new_mode):
        self.mode = new_mode
        self.counter = 0
        self.stage = "EXTENDED"
        self.best_flexion = 180.0

    def update(self, angle):
        if self.mode == "ARM":
            flexion_threshold = 70.0
            extension_threshold = 150.0
        else:
            flexion_threshold = 100.0
            extension_threshold = 160.0

        if angle >= extension_threshold:
            if self.stage == "FLEXED":
                self.counter += 1
            self.stage = "EXTENDED"

        elif angle <= flexion_threshold:
            self.stage = "FLEXED"
            if angle < self.best_flexion:
                self.best_flexion = angle

        return self.counter, self.stage, self.best_flexion