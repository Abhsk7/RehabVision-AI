"""
RehabVision - Visual Skeleton Renderer
Draws joints, connects skeletal lines, and overlays biomechanical feedback.
"""

import cv2
import numpy as np
from src.biomechanics import calculate_joint_angle, classify_knee_posture

# 1. Create a black digital image canvas (Height=500, Width=500, 3 color channels BGR)
canvas = np.zeros((500, 500, 3), dtype=np.uint8)

# 2. Define coordinates for Hip, Knee, and Ankle (Simulating a squat)
hip = (150, 150)
knee = (250, 250)
ankle = (200, 400)

# 3. Calculate angle using our biomechanics module
angle = calculate_joint_angle(hip, knee, ankle)
posture = classify_knee_posture(angle)

# 4. Draw skeletal lines (Yellow: BGR = (0, 255, 255), thickness = 3)
cv2.line(canvas, hip, knee, (0, 255, 255), 3)
cv2.line(canvas, knee, ankle, (0, 255, 255), 3)

# 5. Draw joint circles (Green: BGR = (0, 255, 0), radius = 8, filled = -1)
cv2.circle(canvas, hip, 8, (0, 255, 0), -1)
cv2.circle(canvas, knee, 8, (0, 255, 0), -1)
cv2.circle(canvas, ankle, 8, (0, 255, 0), -1)

# 6. Add text labels next to the joints
cv2.putText(canvas, "Hip", (hip[0] - 50, hip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
cv2.putText(canvas, "Knee", (knee[0] + 15, knee[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
cv2.putText(canvas, "Ankle", (ankle[0] - 60, ankle[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# 7. Overlay the Biomechanical Feedback at the top
cv2.putText(canvas, f"Knee Angle: {angle} deg", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
cv2.putText(canvas, f"Posture: {posture}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# 8. Save the output image
output_path = "rehab_sample.png"
cv2.imwrite(output_path, canvas)
print(f"Visual skeleton successfully saved to {output_path}!")