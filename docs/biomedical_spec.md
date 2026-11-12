# RehabVision: Biomedical Engineering Specifications

## 1. Project Overview
- **Project Name:** RehabVision
- **Focus Areas:** Multi-Joint Kinematics & Real-Time Rehabilitation Monitoring
- **Target Movements:**
  1. Upper-Limb: Elbow Flexion & Extension (Desk/Arm Therapy)
  2. Lower-Limb: Knee Flexion & Extension (Squat & Gait Therapy)

---

## 2. Anatomical Landmarks & Joint Kinematics

### A. Upper-Limb (Arm / Elbow Rehabilitation)
* **Proximal Joint (Shoulder):** Acromion / Glenohumeral Joint (MediaPipe ID: 11 Left / 12 Right)
* **Vertex Joint (Elbow):** Lateral Epicondyle of Humerus (MediaPipe ID: 13 Left / 14 Right)
* **Distal Joint (Wrist):** Radial/Ulnar Styloid Process (MediaPipe ID: 15 Left / 16 Right)

**Range of Motion (ROM) Criteria:**
* **Arm Straight / Full Extension:** 155° – 180°
* **Half Flexion (Functional 90° Bend):** 75° – 154°
* **Full Flexion (Bicep Curl / Hand to Shoulder):** < 75°

---

### B. Lower-Limb (Leg / Knee Rehabilitation)
* **Proximal Joint (Hip):** Greater Trochanter of the Femur (MediaPipe ID: 23 Left / 24 Right)
* **Vertex Joint (Knee):** Lateral Epicondyle of the Femur (MediaPipe ID: 25 Left / 26 Right)
* **Distal Joint (Ankle):** Lateral Malleolus of the Fibula (MediaPipe ID: 27 Left / 28 Right)

**Range of Motion (ROM) Criteria:**
* **Standing Straight / Full Extension:** 165° – 180°
* **Quarter Squat (Gentle Rehab):** 135° – 164°
* **Half Squat (Functional Strength):** 90° – 134°
* **Deep Squat (High Flexion):** < 90°

---

## 3. Signal Processing & Biomedical Noise Filtering
* **Sensor Jitter:** Raw camera pixel predictions fluctuate at 30 fps due to sensor noise and lighting variations.
* **Filter Applied:** Exponential Moving Average (EMA) with smoothing factor $\alpha = 0.35$.
* **Formula:** $\theta_{\text{smooth}} = \alpha \cdot \theta_{\text{raw}} + (1 - \alpha) \cdot \theta_{\text{previous}}$

---

## 4. Safety & Clinical Disclaimers
* Measurements provided by RehabVision are computer vision-based kinematic estimates for exercise feedback and posture guidance.
* This system is experimental software and is not intended for formal clinical diagnosis or invasive medical decisions without clinical calibration.