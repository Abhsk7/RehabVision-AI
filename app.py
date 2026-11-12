"""
RehabVision AI - Universal Web Server & Biomechanics Monitor
Cloud-Ready for Render with Graceful Headless Fallback.
"""

import io
import os
import sys
import atexit
import signal
import threading
import webbrowser
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, send_file
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

app = Flask(__name__)

# Graceful AI Engine Loader (Works on Mac local AND Linux Headless Cloud)
detector = None
try:
    from src.pose_detector import PoseDetector
    from src.biomechanics import (
        calculate_joint_angle,
        classify_knee_posture,
        classify_elbow_posture,
        AngleSmoother,
        ExerciseTracker
    )
    detector = PoseDetector()
    smoother = AngleSmoother(alpha=0.35)
    tracker = ExerciseTracker(mode="ARM")
    print("[INFO] AI Biomechanics Engine Loaded Successfully.")
except Exception as e:
    print(f"[NOTICE] Running in Headless Cloud Mode (Render): {e}")
    smoother = None
    tracker = None

current_mode = "ARM"

telemetry = {
    "reps": 0,
    "angle": 0.0,
    "stage": "EXTENDED",
    "best_rom": 180.0,
    "posture": "Cloud Demo Active",
    "mode": "ARM",
    "side": "Active"
}

def get_camera():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        return cap if cap.isOpened() else None
    except Exception:
        return None

camera = get_camera()

def cleanup(*args):
    if camera and camera.isOpened():
        camera.release()
    cv2.destroyAllWindows()
    sys.exit(0)

atexit.register(cleanup)
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def generate_frames():
    global current_mode, telemetry

    while True:
        # 1. Local Machine with Camera
        if camera and camera.isOpened() and detector:
            success, frame = camera.read()
            if not success:
                continue

            h, w, _ = frame.shape
            landmarks = detector.process_frame(frame)

            if landmarks:
                for (p1_id, p2_id) in detector.connections:
                    pt1 = landmarks[p1_id]
                    pt2 = landmarks[p2_id]
                    if pt1.visibility > 0.4 and pt2.visibility > 0.4:
                        c1 = (int(pt1.x * w), int(pt1.y * h))
                        c2 = (int(pt2.x * w), int(pt2.y * h))
                        cv2.line(frame, c1, c2, (255, 200, 100), 2)

                for lm in landmarks:
                    if lm.visibility > 0.4:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                if current_mode == "ARM":
                    joint_data = detector.get_best_arm_landmarks(landmarks, w, h)
                    if joint_data:
                        p1, p2, p3 = joint_data["shoulder"], joint_data["elbow"], joint_data["wrist"]
                        side = joint_data["side"]
                        cv2.line(frame, p1, p2, (0, 255, 255), 4)
                        cv2.line(frame, p2, p3, (0, 255, 255), 4)
                        cv2.circle(frame, p1, 8, (0, 0, 255), -1)
                        cv2.circle(frame, p2, 8, (0, 255, 0), -1)
                        cv2.circle(frame, p3, 8, (255, 0, 0), -1)

                        raw_angle = calculate_joint_angle(p1, p2, p3)
                        smooth_angle = smoother.update(raw_angle)
                        posture = classify_elbow_posture(smooth_angle)
                        reps, stage, best_rom = tracker.update(smooth_angle)

                        telemetry = {
                            "reps": reps,
                            "angle": smooth_angle,
                            "stage": stage,
                            "best_rom": best_rom,
                            "posture": posture,
                            "mode": "ARM",
                            "side": side
                        }
                else:
                    joint_data = detector.get_best_knee_landmarks(landmarks, w, h)
                    if joint_data:
                        p1, p2, p3 = joint_data["hip"], joint_data["knee"], joint_data["ankle"]
                        side = joint_data["side"]
                        cv2.line(frame, p1, p2, (0, 255, 255), 4)
                        cv2.line(frame, p2, p3, (0, 255, 255), 4)
                        cv2.circle(frame, p1, 8, (0, 0, 255), -1)
                        cv2.circle(frame, p2, 8, (0, 255, 0), -1)
                        cv2.circle(frame, p3, 8, (255, 0, 0), -1)

                        raw_angle = calculate_joint_angle(p1, p2, p3)
                        smooth_angle = smoother.update(raw_angle)
                        posture = classify_knee_posture(smooth_angle)
                        reps, stage, best_rom = tracker.update(smooth_angle)

                        telemetry = {
                            "reps": reps,
                            "angle": smooth_angle,
                            "stage": stage,
                            "best_rom": best_rom,
                            "posture": posture,
                            "mode": "KNEE",
                            "side": side
                        }
        else:
            # 2. Cloud Server Demo Canvas (Render Data Center)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "REHABVISION AI", (180, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            cv2.putText(frame, "Cloud Server Live - 24/7 Deployment", (90, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Biomedical Kinematics Engine Ready", (110, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, "Click Export PDF below for Clinical Report", (80, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def api_stats():
    return jsonify(telemetry)

@app.route('/api/set_mode/<mode>')
def set_mode(mode):
    global current_mode
    if mode in ["ARM", "KNEE"]:
        current_mode = mode
        if smoother and tracker:
            smoother.smoothed_angle = None
            tracker.set_mode(mode)
    return jsonify({"status": "success", "mode": current_mode})

@app.route('/api/reset')
def reset_session():
    if tracker:
        tracker.counter = 0
        tracker.best_flexion = 180.0
        tracker.stage = "EXTENDED"
    return jsonify({"status": "reset"})

@app.route('/download_report')
def download_report():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor('#0f172a'), spaceAfter=4, fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'Subtitle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#64748b'), spaceAfter=14
    )
    section_style = ParagraphStyle(
        'Section', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#1e293b'), spaceBefore=12, spaceAfter=8, fontName='Helvetica-Bold'
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#334155'), leading=14
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#94a3b8'), leading=11
    )

    story.append(Paragraph("RehabVision AI", title_style))
    story.append(Paragraph("Patient Rehabilitation Kinematics & Performance Summary", subtitle_style))
    story.append(Spacer(1, 8))

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_info = [
        [Paragraph("<b>Patient ID:</b> PT-2026-0891", body_style), Paragraph(f"<b>Session Date:</b> {now}", body_style)],
        [Paragraph(f"<b>Target Joint:</b> {telemetry['mode']} Rehabilitation", body_style), Paragraph(f"<b>Tracked Side:</b> {telemetry.get('side', 'Detected Side')}", body_style)]
    ]
    t_info = Table(session_info, colWidths=[260, 260])
    t_info.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_info)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Biomechanical Telemetry & Progress", section_style))
    rom_target = "< 70° (Full Flexion)" if telemetry['mode'] == 'ARM' else "< 100° (Functional Squat)"
    metrics_data = [
        ["Metric Description", "Recorded Value", "Clinical Reference Target"],
        ["Total Completed Repetitions", str(telemetry['reps']), "Prescribed: 10 - 15 reps"],
        ["Deepest Range of Motion (ROM)", f"{telemetry['best_rom']}°", rom_target],
        ["Final Joint Angle", f"{telemetry['angle']}°", "Full Extension: 160° - 180°"],
        ["Observed Posture Assessment", telemetry['posture'], "Optimal Kinetic Alignment"],
    ]
    t_metrics = Table(metrics_data, colWidths=[200, 160, 160])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 24))

    story.append(Paragraph("Biomedical Research & Safety Disclaimer", section_style))
    disclaimer_text = (
        "<b>IMPORTANT NOTICE:</b> RehabVision AI is an educational engineering research prototype. "
        "The kinematic measurements and joint angles documented herein are derived from single-camera 2D monocular "
        "optical pose estimation and may contain planar projection deviations. This report does NOT constitute a "
        "certified clinical diagnosis, nor does it replace professional physiotherapy assessment or clinical goniometry. "
        "All rehabilitation exercises must be supervised by a licensed healthcare professional."
    )
    t_disc = Table([[Paragraph(disclaimer_text, disclaimer_style)]], colWidths=[520])
    t_disc.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fffbeb')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#fde68a')),
        ('PADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t_disc)

    doc.build(story)
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"RehabVision_{telemetry['mode']}_Report.pdf",
        mimetype='application/pdf'
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=False)