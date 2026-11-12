# RehabVision AI
### AI-Powered Full-Body Physiotherapy & Rehabilitation Monitoring System

RehabVision AI is an educational software prototype and computer vision system that monitors physical rehabilitation exercises using real-time pose estimation. 

---

## 🌐 Live Website
The application is deployed 24/7 on the cloud:
👉 **[https://rehabvision-ai.onrender.com](https://rehabvision-ai.onrender.com)**

---

## ✨ Features
- **Real-Time Pose Estimation:** Tracks 33 full-body landmarks using Google MediaPipe Pose.
- **Joint Angle Kinematics:** Computes dynamic 2D joint angles using vector mathematics.
- **Noise Reduction:** Filters out camera jitter using an Exponential Moving Average (EMA) filter.
- **Repetition Counter:** Automatically counts complete exercise repetitions using movement phase detection.
- **Exercise Modes:** Supports both Upper-Limb (Bicep Curl) and Lower-Limb (Knee Extension / Squat) rehabilitation.
- **Clinical PDF Export:** Generates an executive medical PDF report summarizing workout metrics and range of motion.

---

## 🛠️ Technology Stack
- **AI & Computer Vision:** Python, OpenCV, MediaPipe
- **Web Backend:** Flask, Gunicorn
- **Reporting:** ReportLab (PDF generation)
- **Deployment:** Render Cloud, GitHub

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abhsk7/RehabVision-AI.git
   cd RehabVision-AI
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the application:**
   ```bash
   python app.py
   ```
   *Your browser will automatically open to `http://127.0.0.1:5001`!*

---

## ⚠️ Medical Disclaimer
This software is an educational student research prototype. It is NOT a certified medical device and does not provide clinical diagnoses. Always consult a licensed physiotherapist before undertaking rehabilitation exercises.

---

## 👥 Authors
- **Abhishek Krishna S** — Software Engineering, AI & Web Architecture
- **Ananya K S** — Biomedical Research & Clinical Biomechanics