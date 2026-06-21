# 🏥 REHATRACK PRO
**AI-Powered Biomechanics & Posture Assessment System**

Rehatrack Pro is a clinical posture and movement screening tool built with Python, OpenCV, and Google MediaPipe. It digitizes the NASM (National Academy of Sports Medicine) Corrective Exercise continuum (CEx) by converting standard 2D webcam feeds into real-time 3D biomechanical analysis.

## 🚀 Features
* **Real-time Tracking:** 33 anatomical landmarks extraction using overclocked MediaPipe (Complexity Level 2).
* **Anti-Jitter Algorithm:** Integrated Exponential Moving Average (EMA) filter to mitigate occlusion errors during deep squats.
* **NASM CEx Protocols:** Automatically assesses Valgus, Varus, Forward Head Posture (FHP), and asymmetrical shifts.
* **Clinical Export:** Generates standardized JSON EMR reports and diagnostic evidence images.

## ⚙️ Installation
1. Clone this repository:
```bash
   git clone [https://github.com/vietnghien/Rehatrack-Pro.git](https://github.com/vietnghien/Rehatrack-Pro.git)
2. Install the required dependencies:
pip install opencv-python mediapipe numpy pillow
3. Run the application:
python main.py
👨‍💻 Authors
Nguyễn Quốc Việt (Hanoi University of Science and Technology)

Ngô Minh Phú (Hanoi University of Science and Technology)
