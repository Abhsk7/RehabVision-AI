"""
RehabVision - Test Runner
Simulates patient joint data and verifies our biomechanics calculations.
"""

from src.biomechanics import calculate_joint_angle, classify_knee_posture

print("==================================================")
print("       REHABVISION: BIOMECHANICS TEST RUN         ")
print("==================================================")

# Test Case 1: Patient standing straight (legs aligned vertically)
hip_standing = (100, 100)
knee_standing = (100, 200)
ankle_standing = (100, 300)

angle_standing = calculate_joint_angle(hip_standing, knee_standing, ankle_standing)
posture_standing = classify_knee_posture(angle_standing)

print(f"\n[Test 1: Standing Position]")
print(f"Coordinates: Hip={hip_standing}, Knee={knee_standing}, Ankle={ankle_standing}")
print(f"Calculated Angle: {angle_standing}°")
print(f"Clinical Posture: {posture_standing}")

# Test Case 2: Patient bending their knee into a 90-degree squat
hip_squat = (50, 200)
knee_squat = (150, 200)
ankle_squat = (150, 300)

angle_squat = calculate_joint_angle(hip_squat, knee_squat, ankle_squat)
posture_squat = classify_knee_posture(angle_squat)

print(f"\n[Test 2: Squat Position]")
print(f"Coordinates: Hip={hip_squat}, Knee={knee_squat}, Ankle={ankle_squat}")
print(f"Calculated Angle: {angle_squat}°")
print(f"Clinical Posture: {posture_squat}")
print("==================================================")