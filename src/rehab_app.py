"""
RehabVision - Live Multi-Joint Rehabilitation Tracker
Includes Real-Time Repetition Counting and Range-of-Motion (ROM) Analytics.
"""

import cv2
from src.pose_detector import PoseDetector
from src.biomechanics import (
    calculate_joint_angle,
    classify_knee_posture,
    classify_elbow_posture,
    AngleSmoother,
    ExerciseTracker
)

def main():
    print("==================================================")
    print("      REHABVISION: EXERCISE TRACKER ACTIVE        ")
    print("==================================================")

    # Cross-platform camera capture (compatible with macOS and Windows)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[ERROR] Could not access webcam.")
        return

    detector = PoseDetector()
    smoother = AngleSmoother(alpha=0.35)
    
    # Initialize our Exercise Repetition Tracker
    mode = "ARM"
    tracker = ExerciseTracker(mode=mode)

    print("[CONTROLS]")
    print("Press 'm' to switch between ARM and KNEE mode.")
    print("Press 'q' to quit and see your session summary.")
    print("==================================================")

    while True:
        success, frame = cap.read()
        if not success:
            break

        h, w, _ = frame.shape

        # 1. Detect AI landmarks
        landmarks = detector.process_frame(frame)

        if landmarks:
            # 2. Draw Skeleton lines
            for (p1_id, p2_id) in detector.connections:
                pt1 = landmarks[p1_id]
                pt2 = landmarks[p2_id]
                if pt1.visibility > 0.4 and pt2.visibility > 0.4:
                    c1 = (int(pt1.x * w), int(pt1.y * h))
                    c2 = (int(pt2.x * w), int(pt2.y * h))
                    cv2.line(frame, c1, c2, (255, 200, 100), 2)

            # 3. Draw Landmark Dots
            for lm in landmarks:
                if lm.visibility > 0.4:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # ==========================================
            # ACTIVE REHABILITATION MODE
            # ==========================================
            if mode == "ARM":
                joint_data = detector.get_best_arm_landmarks(landmarks, w, h)
                if joint_data:
                    p1, p2, p3 = joint_data["shoulder"], joint_data["elbow"], joint_data["wrist"]
                    side = joint_data["side"]

                    # Highlight arm
                    cv2.line(frame, p1, p2, (0, 255, 255), 4)
                    cv2.line(frame, p2, p3, (0, 255, 255), 4)
                    cv2.circle(frame, p1, 9, (0, 0, 255), -1)
                    cv2.circle(frame, p2, 9, (0, 255, 0), -1)
                    cv2.circle(frame, p3, 9, (255, 0, 0), -1)

                    raw_angle = calculate_joint_angle(p1, p2, p3)
                    smooth_angle = smoother.update(raw_angle)
                    posture = classify_elbow_posture(smooth_angle)
                    reps, stage, best_rom = tracker.update(smooth_angle)

                    # Top Left: Clinical Posture HUD
                    cv2.putText(frame, "MODE: ARM REHAB", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                    cv2.putText(frame, f"Tracked: {side} Arm", (30, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                    cv2.putText(frame, f"Angle: {smooth_angle} deg", (30, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, f"Posture: {posture}", (30, 145),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Top Right: Exercise Analytics Box
                    cv2.rectangle(frame, (w - 280, 20), (w - 20, 160), (30, 30, 30), -1)
                    cv2.rectangle(frame, (w - 280, 20), (w - 20, 160), (0, 255, 255), 2)
                    cv2.putText(frame, f"REPS: {reps}", (w - 260, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(frame, f"Stage: {stage}", (w - 260, 105),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Best ROM: {best_rom} deg", (w - 260, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            else:
                joint_data = detector.get_best_knee_landmarks(landmarks, w, h)
                if joint_data:
                    p1, p2, p3 = joint_data["hip"], joint_data["knee"], joint_data["ankle"]
                    side = joint_data["side"]

                    cv2.line(frame, p1, p2, (0, 255, 255), 4)
                    cv2.line(frame, p2, p3, (0, 255, 255), 4)
                    cv2.circle(frame, p1, 9, (0, 0, 255), -1)
                    cv2.circle(frame, p2, 9, (0, 255, 0), -1)
                    cv2.circle(frame, p3, 9, (255, 0, 0), -1)

                    raw_angle = calculate_joint_angle(p1, p2, p3)
                    smooth_angle = smoother.update(raw_angle)
                    posture = classify_knee_posture(smooth_angle)
                    reps, stage, best_rom = tracker.update(smooth_angle)

                    cv2.putText(frame, "MODE: KNEE REHAB", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
                    cv2.putText(frame, f"Tracked: {side} Leg", (30, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                    cv2.putText(frame, f"Angle: {smooth_angle} deg", (30, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, f"Posture: {posture}", (30, 145),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Top Right: Exercise Analytics Box
                    cv2.rectangle(frame, (w - 280, 20), (w - 20, 160), (30, 30, 30), -1)
                    cv2.rectangle(frame, (w - 280, 20), (w - 20, 160), (0, 100, 255), 2)
                    cv2.putText(frame, f"REPS: {reps}", (w - 260, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(frame, f"Stage: {stage}", (w - 260, 105),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Best ROM: {best_rom} deg", (w - 260, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.putText(frame, "Press 'm' to Switch Mode (Arm/Knee) | 'q' to Quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("RehabVision - Live Multi-Joint Rehabilitation Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
            mode = "KNEE" if mode == "ARM" else "ARM"
            smoother.smoothed_angle = None
            tracker.set_mode(mode)
            print(f"Switched mode to: {mode}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n==================================================")
    print("               SESSION COMPLETED                  ")
    print(f"Total Repetitions: {tracker.counter}")
    print(f"Deepest Range of Motion (ROM): {tracker.best_flexion}°")
    print("==================================================")

if __name__ == "__main__":
    main()