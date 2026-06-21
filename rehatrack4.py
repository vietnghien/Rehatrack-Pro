import os
import warnings

# Bịt mồm toàn bộ log rác của TensorFlow & MediaPipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['GLOG_minloglevel'] = '2'      
warnings.filterwarnings('ignore')         

import cv2
import numpy as np
import json
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk

import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing

# --- 1. HỆ TOÁN HỌC & GIẢI PHẪU ---
def check_shoulder_tilt(l_sh, r_sh):
    dx = r_sh[0] - l_sh[0]
    dy = r_sh[1] - l_sh[1]
    angle = np.abs(np.arctan2(dy, dx) * 180.0 / np.pi)
    tilt = round(angle if angle < 90 else 180 - angle, 1)
    if tilt <= 1.5: return tilt, "Vai cân bằng", [], []
    elif l_sh[1] > r_sh[1]: return tilt, "Vai Trái thấp", ["trap_r", "lat_l"], ["trap_low_l"]
    else: return tilt, "Vai Phải thấp", ["trap_l", "lat_r"], ["trap_low_r"]

def check_hip_tilt(l_hip, r_hip):
    dx = r_hip[0] - l_hip[0]
    dy = r_hip[1] - l_hip[1]
    angle = np.abs(np.arctan2(dy, dx) * 180.0 / np.pi)
    tilt = round(angle if angle < 90 else 180 - angle, 1)
    if tilt <= 1.5: return tilt, "Hông cân bằng", [], []
    elif l_hip[1] > r_hip[1]: return tilt, "Hông Trái thấp", ["ql_r"], ["glute_med_r"]
    else: return tilt, "Hông Phải thấp", ["ql_l"], ["glute_med_l"]

def check_forward_head(ear, shoulder):
    dx = abs(ear[0] - shoulder[0])
    if dx > 0.045: return "Cổ Rùa (FHP)", ["trap_r", "trap_l", "pec"], ["deep_neck"]
    return "Cổ chuẩn", [], []

def check_knee_valgus(l_knee, r_knee, l_ankle, r_ankle):
    knee_dist = np.sqrt((l_knee[0] - r_knee[0])**2 + (l_knee[1] - r_knee[1])**2)
    ankle_dist = np.sqrt((l_ankle[0] - r_ankle[0])**2 + (l_ankle[1] - r_ankle[1])**2)
    if ankle_dist == 0: return "Đang quét...", [], []
    ratio = knee_dist / ankle_dist
    if ratio < 0.85: return "Sụp gối (Valgus)", ["adductors"], ["glute_med_l", "glute_med_r"]
    elif ratio > 1.3: return "Vòng kiềng (Varus)", ["it_band_l", "it_band_r"], ["adductors"]
    return "Trục chuẩn", [], []

def check_sagittal_squat(shoulder, hip, wrist):
    dx_torso = abs(shoulder[0] - hip[0])
    dy_torso = abs(shoulder[1] - hip[1])
    torso_angle = np.abs(np.arctan2(dx_torso, dy_torso) * 180.0 / np.pi)
    if wrist[1] > shoulder[1] + 0.1: return "Tay đổ về trước", ["lat_l", "lat_r", "pec"], ["trap_low_l", "trap_low_r"]
    elif torso_angle > 45: return "Võng lưng", ["psoas"], ["glute_max"]
    return "Trục lưng chuẩn", [], []

MUSCLE_DICT = {
    "trap_r": "Thang trên (Phải)", "trap_l": "Thang trên (Trái)",
    "lat_r": "Cơ Xô (Phải)", "lat_l": "Cơ Xô (Trái)",
    "trap_low_r": "Thang dưới (Phải)", "trap_low_l": "Thang dưới (Trái)",
    "ql_r": "Vuông thắt lưng (Phải)", "ql_l": "Vuông thắt lưng (Trái)",
    "glute_med_r": "Mông nhỡ (Phải)", "glute_med_l": "Mông nhỡ (Trái)",
    "pec": "Cơ ngực nhỏ/lớn", "deep_neck": "Cơ gập cổ sâu",
    "adductors": "Cơ khép đùi", "it_band_l": "Dải chậu chày (Trái)", "it_band_r": "Dải chậu chày (Phải)",
    "psoas": "Cơ gập hông (Psoas)", "glute_max": "Cơ mông lớn"
}

CEX_PROTOCOL = {
    "Sụp gối (Valgus)": {
        "1_Inhibit": "Lăn bọt Cơ khép & Dải chậu chày (IT Band) 30-60s.",
        "2_Lengthen": "Kéo giãn tĩnh Cơ khép (Kneeling Adductor Stretch) 30s.",
        "3_Activate": "Tập Clamshells hoặc đi bộ cua với dây kháng lực.",
        "4_Integrate": "Step-up có kiểm soát trục đầu gối."
    },
    "Vòng kiềng (Varus)": {
        "1_Inhibit": "Lăn bọt Dải chậu chày (IT Band) & Cơ hình lê.",
        "2_Lengthen": "Kéo giãn IT Band.",
        "3_Activate": "Kích hoạt cơ Khép đùi (Copenhagen Plank).",
        "4_Integrate": "Squat kiểm soát bóng kẹp giữa 2 đùi."
    },
    "Cổ Rùa (FHP)": {
        "1_Inhibit": "Massage điểm Cơ Thang trên & Ức đòn chũm 30-60s.",
        "2_Lengthen": "Kéo giãn Cơ Ngực ở khung cửa (Doorway Stretch) 30s.",
        "3_Activate": "Tập gập cổ sâu (Chin Tucks) 10-15 reps.",
        "4_Integrate": "Tập kéo cáp mặt (Face Pulls) giữ thẳng trục cổ."
    },
    "Tay đổ về trước": {
        "1_Inhibit": "Lăn bọt Cơ Xô (Latissimus Dorsi) & Cơ ngực.",
        "2_Lengthen": "Kéo giãn Cơ Xô trên ghế 30s.",
        "3_Activate": "Tập Y-W-T raises nằm sấp để kích hoạt Thang dưới.",
        "4_Integrate": "Squat to Row (Squat kết hợp kéo cáp ngang)."
    },
    "Võng lưng": {
        "1_Inhibit": "Lăn bọt đùi trước (Quads). Tránh tự lăn vùng thắt lưng sâu.",
        "2_Lengthen": "Kéo giãn cơ gập hông (Kneeling Hip Flexor Stretch) 30s.",
        "3_Activate": "Kích hoạt Cơ mông lớn (Glute Bridges) & Core (Plank).",
        "4_Integrate": "Wall Squat tựa bóng, siết chặt cơ bụng."
    },
    "Lệch Vai": {
        "1_Inhibit": "Lăn bọt Cơ Xô bên vai bị sụp & Cơ Thang trên bên vai nhô cao.",
        "2_Lengthen": "Kéo giãn tĩnh Cơ Xô & Cơ Thang.",
        "3_Activate": "Kích hoạt Cơ Thang dưới (Lower Trap).",
        "4_Integrate": "Kéo cáp một tay (Single-arm Row) cân bằng 2 bên."
    },
    "Lệch Hông": {
        "1_Inhibit": "Massage Cơ Vuông thắt lưng (QL) bên hông bị nhô cao.",
        "2_Lengthen": "Kéo giãn QL và lườn.",
        "3_Activate": "Kích hoạt Cơ mông nhỡ (Glute Med) bên hông bị sụp thấp.",
        "4_Integrate": "Single-leg Deadlift (RDL 1 chân) có hỗ trợ thăng bằng."
    }
}

PROTOCOL = [
    {"id": "STATIC_FRONT", "name": "1. Đánh giá tĩnh (Mặt trước)", "angle": "GÓC MÁY TRỰC DIỆN", "instruction": "Camera quay thẳng mặt. Bệnh nhân đứng thẳng.", "type": "static_front"},
    {"id": "STATIC_SIDE", "name": "2. Đánh giá tĩnh (Mặt bên)", "angle": "GÓC MÁY MẶT BÊN", "instruction": "Xoay người 90 độ. Camera quay ngang.", "type": "static_side"},
    {"id": "OHSA_FRONT", "name": "3. OH Squat (Mặt trước)", "angle": "GÓC MÁY TRỰC DIỆN", "instruction": "Quay mặt vào Camera. Giơ tay, Squat.", "type": "ohsa_front"},
    {"id": "OHSA_SIDE", "name": "4. OH Squat (Mặt bên)", "angle": "GÓC MÁY MẶT BÊN", "instruction": "Xoay ngang. Giơ tay, Squat.", "type": "ohsa_side"}
]

# --- 2. GIAO DIỆN HỆ THỐNG ---
class RehatrackApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1200x750")
        self.window.configure(bg="#1e272e")
        try: self.window.state('zoomed') 
        except: pass

        self.current_step = 0
        self.is_running = False
        self.cap = None
        self.export_frame = None 
        
        self.final_report = {
            "Clinic_Info": "Rehatrack Biomechanics System",
            "Patient_ID": "NASM_CLIENT_01", 
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            "Assessments": {}, 
            "CEx_Prescription": {}
        }
        
        self.mp_pose = mp_pose 
        self.mp_drawing = mp_drawing 
        
        # --- BẢN UPDATE: TỐI ƯU HÓA LÕI AI MEDIAPIPE LÊN LEVEL 2 ---
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,             # Ép xung độ chính xác tối đa
            smooth_landmarks=True,          # Kích hoạt bộ lọc mượt gốc
            min_detection_confidence=0.75,  # Tăng ngưỡng nhận diện
            min_tracking_confidence=0.75    # Tăng ngưỡng bám sát
        )
        
        # --- BẢN UPDATE: BIẾN CHO BỘ LỌC CHỐNG RUNG EMA ---
        self.prev_landmarks = {}
        self.alpha_ema = 0.6  # Hệ số làm mượt (ngọt nước nhất cho Squat)

        self.main_container = tk.Frame(window, bg="#1e272e")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # UI Trái
        self.video_frame = tk.Frame(self.main_container, bg="black", bd=2, relief=tk.SUNKEN)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.guide_frame = tk.Frame(self.video_frame, bg="#e1b12c", pady=5)
        self.guide_frame.pack(fill=tk.X)
        self.lbl_angle = tk.Label(self.guide_frame, text="GÓC MÁY...", font=("Helvetica", 14, "bold"), fg="#1e272e", bg="#e1b12c")
        self.lbl_angle.pack()
        self.lbl_instruction = tk.Label(self.guide_frame, text="Hướng dẫn...", font=("Helvetica", 11, "italic"), fg="#1e272e", bg="#e1b12c")
        self.lbl_instruction.pack()

        self.canvas_container = tk.Frame(self.video_frame, bg="black")
        self.canvas_container.pack(expand=True)
        self.canvas = tk.Canvas(self.canvas_container, width=640, height=480, bg="black", highlightthickness=0)
        self.canvas.pack()

        # UI Phải
        self.control_frame = tk.Frame(self.main_container, bg="#1e272e", width=420)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        self.control_frame.pack_propagate(False) 

        tk.Label(self.control_frame, text="REHATRACK PRO", font=("Helvetica", 22, "bold"), fg="#0fb9b1", bg="#1e272e").pack()
        self.lbl_status = tk.Label(self.control_frame, text="SẴN SÀNG", font=("Helvetica", 11, "bold"), fg="#f7b731", bg="#1e272e")
        self.lbl_status.pack(pady=2)

        btn_frame = tk.Frame(self.control_frame, bg="#1e272e")
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        self.btn_start = tk.Button(btn_frame, text="▶ BẬT CAMERA", font=("Helvetica", 11, "bold"), bg="#20bf6b", fg="white", command=self.start_camera, height=2)
        self.btn_start.pack(fill=tk.X, pady=2)
        self.btn_stop = tk.Button(btn_frame, text="⏹ TẮT CAMERA", font=("Helvetica", 11, "bold"), bg="#f39c12", fg="white", command=self.stop_camera, height=2, state=tk.DISABLED)
        self.btn_stop.pack(fill=tk.X, pady=2)
        self.btn_next = tk.Button(btn_frame, text="📸 LƯU ẢNH & CHUYỂN BÀI", font=("Helvetica", 11, "bold"), bg="#45aaf2", fg="white", command=self.next_joint, height=2, state=tk.DISABLED)
        self.btn_next.pack(fill=tk.X, pady=2)
        self.btn_export = tk.Button(btn_frame, text="💾 XUẤT HỒ SƠ Y KHOA (JSON)", font=("Helvetica", 11, "bold"), bg="#eb3b5a", fg="white", command=self.export_report, height=2, state=tk.DISABLED)
        self.btn_export.pack(fill=tk.X, pady=2)

        config_frame = tk.Frame(self.control_frame, bg="#1e272e")
        config_frame.pack(fill=tk.X, pady=5)
        tk.Label(config_frame, text="1. NGUỒN CAMERA:", font=("Helvetica", 9, "bold"), fg="#a5b1c2", bg="#1e272e").pack(pady=(2, 0))
        self.cam_selector = ttk.Combobox(config_frame, values=["Webcam (0)", "Cam Rời (1)", "📱 IP Camera"], state="readonly", font=("Helvetica", 10))
        self.cam_selector.current(0)
        self.cam_selector.pack(fill=tk.X)
        tk.Label(config_frame, text="2. GÓC XOAY:", font=("Helvetica", 9, "bold"), fg="#a5b1c2", bg="#1e272e").pack(pady=(5, 0))
        self.rotate_selector = ttk.Combobox(config_frame, values=["Chuẩn 0°", "Dọc Phải 90°", "Dọc Trái 90°", "Ngược 180°"], state="readonly", font=("Helvetica", 10))
        self.rotate_selector.current(0)
        self.rotate_selector.pack(fill=tk.X)

        self.data_frame = tk.Frame(self.control_frame, bg="#2d3436", bd=1, relief=tk.RIDGE)
        self.data_frame.pack(fill=tk.X, pady=5, ipady=2)
        self.lbl_data_1 = tk.Label(self.data_frame, text="--", font=("Consolas", 12, "bold"), fg="#20bf6b", bg="#2d3436")
        self.lbl_data_1.pack()
        self.lbl_data_2 = tk.Label(self.data_frame, text="--", font=("Consolas", 12, "bold"), fg="#45aaf2", bg="#2d3436")
        self.lbl_data_2.pack()

        self.anatomy_frame = tk.Frame(self.control_frame, bg="#1e272e")
        self.anatomy_frame.pack(fill=tk.BOTH, expand=True, pady=2) 
        tk.Label(self.anatomy_frame, text="BẢN ĐỒ CƠ BÙ TRỪ", font=("Helvetica", 10, "bold"), fg="#ffb8b8", bg="#1e272e").pack()
        
        self.anatomy_canvas = tk.Canvas(self.anatomy_frame, width=150, height=250, bg="#2d3436", highlightthickness=1, highlightbackground="#45aaf2")
        self.anatomy_canvas.pack(side=tk.LEFT, pady=5)
        self.note_frame = tk.Frame(self.anatomy_frame, bg="#1e272e")
        self.note_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(self.note_frame, text="CHÚ GIẢI:", font=("Helvetica", 9, "bold"), fg="#a5b1c2", bg="#1e272e", anchor="w").pack(fill=tk.X)
        self.lbl_tight = tk.Label(self.note_frame, text="🔴 BÓ/CĂNG:\n--", font=("Consolas", 9), fg="#ff7675", bg="#1e272e", justify=tk.LEFT, wraplength=180)
        self.lbl_tight.pack(pady=2, anchor="w")
        self.lbl_weak = tk.Label(self.note_frame, text="🔵 YẾU/GIÃN:\n--", font=("Consolas", 9), fg="#74b9ff", bg="#1e272e", justify=tk.LEFT, wraplength=180)
        self.lbl_weak.pack(pady=2, anchor="w")

        self.update_ui_labels()
        self.render_avatar([], [])
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.current_metrics = self.create_empty_metrics()

    # --- BẢN UPDATE: HÀM LỌC CHỐNG RUNG (EMA FILTER) ---
    def apply_ema_filter(self, current_landmarks):
        smoothed = {}
        for idx, lm in enumerate(current_landmarks.landmark):
            if lm.visibility > 0.5:
                curr_pt = np.array([lm.x, lm.y])
                if idx not in self.prev_landmarks:
                    self.prev_landmarks[idx] = curr_pt
                    smoothed[idx] = curr_pt
                else:
                    smoothed_pt = self.alpha_ema * curr_pt + (1 - self.alpha_ema) * self.prev_landmarks[idx]
                    self.prev_landmarks[idx] = smoothed_pt
                    smoothed[idx] = smoothed_pt
            else:
                smoothed[idx] = self.prev_landmarks.get(idx, np.array([lm.x, lm.y]))
        return smoothed

    def create_empty_metrics(self):
        return {"Findings": "Đang phân tích...", "Overactive": [], "Underactive": [], "Metrics": {}, "Image_Path": ""}

    def render_avatar(self, tight_ids, weak_ids):
        img = np.ones((250, 150, 3), dtype=np.uint8) * 45 
        color_body = (150, 150, 150)
        
        # Avatar có đầy đủ chân tay cân đối
        cv2.circle(img, (75, 30), 18, color_body, -1) 
        cv2.ellipse(img, (75, 100), (28, 45), 0, 0, 360, color_body, -1) 
        cv2.ellipse(img, (75, 155), (30, 12), 0, 0, 360, color_body, -1) 
        cv2.ellipse(img, (55, 200), (10, 40), 0, 0, 360, color_body, -1) 
        cv2.ellipse(img, (95, 200), (10, 40), 0, 0, 360, color_body, -1) 
        cv2.ellipse(img, (40, 100), (8, 35), 20, 0, 360, color_body, -1) 
        cv2.ellipse(img, (110, 100), (8, 35), -20, 0, 360, color_body, -1) 
        cv2.ellipse(img, (28, 145), (6, 30), 10, 0, 360, color_body, -1) 
        cv2.ellipse(img, (122, 145), (6, 30), -10, 0, 360, color_body, -1)
        
        MUSCLE_COORDS = {"trap_l": (55, 60), "trap_r": (95, 60), "lat_l": (50, 105), "lat_r": (100, 105), "ql_l": (60, 130), "ql_r": (90, 130), "glute_med_l": (45, 155), "glute_med_r": (105, 155), "pec": (75, 75), "adductors": (75, 180), "it_band_l": (45, 195), "it_band_r": (105, 195), "psoas": (75, 145), "glute_max": (75, 165)}
        overlay = img.copy()
        
        for m_id in weak_ids: 
            if m_id in MUSCLE_COORDS: cv2.circle(overlay, MUSCLE_COORDS[m_id], 10, (255, 100, 50), -1)
        for m_id in tight_ids: 
            if m_id in MUSCLE_COORDS: cv2.circle(overlay, MUSCLE_COORDS[m_id], 10, (50, 50, 255), -1)
            
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.avatar_tk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        self.anatomy_canvas.create_image(0, 0, image=self.avatar_tk, anchor=tk.NW)

    def start_camera(self):
        if not self.is_running:
            selected_index = self.cam_selector.current() 
            if selected_index == 2:
                ip_url = simpledialog.askstring("Nhập IP", "Link DroidCam:\n(VD: http://192.168.1.15:4747/video)", parent=self.window)
                if not ip_url: return
                if not ip_url.endswith("/video"): ip_url = ip_url + "video" if ip_url.endswith("/") else ip_url + "/video"
                self.cap = cv2.VideoCapture(ip_url)
            else:
                self.cap = cv2.VideoCapture(selected_index)
            
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", "Không kết nối được Camera!")
                return
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
            self.is_running = True
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.btn_next.config(state=tk.NORMAL)
            self.btn_export.config(state=tk.NORMAL)
            self.current_metrics = self.create_empty_metrics()
            self.update_ui_labels()
            self.process_frame()

    def stop_camera(self):
        self.is_running = False
        if self.cap: self.cap.release()
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.canvas.delete("all")
        self.render_avatar([], [])

    def record_cex(self, findings):
        if "Vai" in findings and "thấp" in findings: self.final_report["CEx_Prescription"]["Lỗi Lệch Vai"] = CEX_PROTOCOL["Lệch Vai"]
        if "Hông" in findings and "thấp" in findings: self.final_report["CEx_Prescription"]["Lỗi Lệch Hông"] = CEX_PROTOCOL["Lệch Hông"]
        if "Cổ Rùa" in findings: self.final_report["CEx_Prescription"]["Lỗi Cổ Rùa"] = CEX_PROTOCOL["Cổ Rùa (FHP)"]
        if "Sụp gối" in findings: self.final_report["CEx_Prescription"]["Lỗi Sụp Gối"] = CEX_PROTOCOL["Sụp gối (Valgus)"]
        if "Vòng kiềng" in findings: self.final_report["CEx_Prescription"]["Lỗi Vòng Kiềng"] = CEX_PROTOCOL["Vòng kiềng (Varus)"]
        if "Tay đổ về trước" in findings: self.final_report["CEx_Prescription"]["Lỗi Tay đổ về trước"] = CEX_PROTOCOL["Tay đổ về trước"]
        if "Võng lưng" in findings: self.final_report["CEx_Prescription"]["Lỗi Võng Lưng"] = CEX_PROTOCOL["Võng lưng"]

    def next_joint(self):
        step_id = PROTOCOL[self.current_step]["id"]
        
        # Tính năng vẫn được giữ lại: Tự động lưu ảnh màn hình minh chứng
        if self.export_frame is not None:
            img_filename = f"Capture_{step_id}_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(img_filename, self.export_frame)
            self.current_metrics["Image_Path"] = img_filename 

        self.final_report["Assessments"][step_id] = self.current_metrics.copy()
        self.record_cex(self.current_metrics["Findings"])

        self.current_step += 1
        if self.current_step >= len(PROTOCOL):
            messagebox.showinfo("Xong", "Đã quét đủ 4 bài! Hãy bấm XUẤT HỒ SƠ Y KHOA.")
            self.stop_camera()
            self.current_step = 0
            self.update_ui_labels()
        else:
            self.current_metrics = self.create_empty_metrics() 
            self.update_ui_labels()
            self.render_avatar([], [])

    def export_report(self):
        step_id = PROTOCOL[self.current_step]["id"]
        self.final_report["Assessments"][step_id] = self.current_metrics.copy()
        self.record_cex(self.current_metrics["Findings"])
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_filename = f"Report_{timestamp}.json"
        
        try:
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(self.final_report, f, ensure_ascii=False, indent=4)
            messagebox.showinfo("Thành công", f"Đã lưu Dữ liệu Y khoa & Ảnh minh chứng vào thư mục!\nFile báo cáo: {json_filename}")
        except Exception as e:
            messagebox.showerror("Lỗi hệ thống", f"Không thể lưu file: {str(e)}")

    def update_ui_labels(self):
        info = PROTOCOL[self.current_step]
        self.lbl_status.config(text=f"BÀI {self.current_step + 1}/4: {info['name']}")
        self.lbl_angle.config(text=info['angle'])
        self.lbl_instruction.config(text=info['instruction'])
        self.lbl_data_1.config(text="--")
        self.lbl_data_2.config(text="--")

    def process_frame(self):
        if self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                rot_idx = self.rotate_selector.current()
                if rot_idx == 1: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rot_idx == 2: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rot_idx == 3: frame = cv2.rotate(frame, cv2.ROTATE_180)

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = image.shape
                
                cv2.line(image, (int(w/2), 0), (int(w/2), h), (255, 255, 0), 2)
                for i in range(0, w, int(w/15)): cv2.line(image, (i, 0), (i, h), (0, 150, 0), 1)
                for j in range(0, h, int(h/15)): cv2.line(image, (0, j), (w, j), (0, 150, 0), 1)

                image.flags.writeable = False
                results = self.pose.process(image)
                image.flags.writeable = True

                all_tight_ids, all_weak_ids = [], []
                tight_texts, weak_texts = [], []

                if self.current_step < len(PROTOCOL):
                    test_info = PROTOCOL[self.current_step]
                    
                    if results.pose_landmarks:
                        # --- BẢN UPDATE: ĐƯA DỮ LIỆU QUA BỘ LỌC CHỐNG RUNG ---
                        smoothed_data = self.apply_ema_filter(results.pose_landmarks)
                        
                        try:
                            if test_info["type"] == "static_front":
                                # Đã thay bằng dữ liệu mượt từ smoothed_data
                                l_sh, r_sh = smoothed_data[11], smoothed_data[12]
                                l_hip, r_hip = smoothed_data[23], smoothed_data[24]
                                s_tilt, s_stat, s_t_id, s_w_id = check_shoulder_tilt(l_sh, r_sh)
                                h_tilt, h_stat, h_t_id, h_w_id = check_hip_tilt(l_hip, r_hip)
                                
                                self.lbl_data_1.config(text=f"Vai: {s_tilt}° ({s_stat})")
                                self.lbl_data_2.config(text=f"Hông: {h_tilt}° ({h_stat})")
                                all_tight_ids, all_weak_ids = s_t_id + h_t_id, s_w_id + h_w_id
                                
                                combined = []
                                if s_stat != "Vai cân bằng": combined.append(s_stat)
                                if h_stat != "Hông cân bằng": combined.append(h_stat)
                                self.current_metrics["Findings"] = " + ".join(combined) if combined else "Cân bằng"
                                
                                cv2.line(image, (int(r_sh[0]*w), int(r_sh[1]*h)), (int(l_sh[0]*w), int(l_sh[1]*h)), (255, 0, 0), 3)
                                cv2.line(image, (int(r_hip[0]*w), int(r_hip[1]*h)), (int(l_hip[0]*w), int(l_hip[1]*h)), (255, 0, 0), 3)

                            elif test_info["type"] == "static_side":
                                stat, t_id, w_id = check_forward_head(smoothed_data[7], smoothed_data[11])
                                self.lbl_data_1.config(text=f"Lỗi: {stat}")
                                all_tight_ids, all_weak_ids = t_id, w_id
                                self.current_metrics["Findings"] = stat
                                cv2.line(image, (int(smoothed_data[11][0]*w), 0), (int(smoothed_data[11][0]*w), h), (255, 0, 255), 2)

                            elif test_info["type"] == "ohsa_front":
                                stat, t_id, w_id = check_knee_valgus(smoothed_data[25], smoothed_data[26], smoothed_data[27], smoothed_data[28])
                                self.lbl_data_1.config(text=f"Lỗi: {stat}")
                                all_tight_ids, all_weak_ids = t_id, w_id
                                self.current_metrics["Findings"] = stat

                            elif test_info["type"] == "ohsa_side":
                                stat, t_id, w_id = check_sagittal_squat(smoothed_data[11], smoothed_data[23], smoothed_data[15])
                                self.lbl_data_1.config(text=f"Lỗi: {stat}")
                                all_tight_ids, all_weak_ids = t_id, w_id
                                self.current_metrics["Findings"] = stat

                            for t_id in all_tight_ids: tight_texts.append(MUSCLE_DICT.get(t_id, t_id))
                            for w_id in all_weak_ids: weak_texts.append(MUSCLE_DICT.get(w_id, w_id))
                            
                            self.current_metrics["Overactive"] = list(set(tight_texts))
                            self.current_metrics["Underactive"] = list(set(weak_texts))
                            
                            self.lbl_tight.config(text="🔴 BÓ:\n" + ("\n".join(set(tight_texts)) if tight_texts else "Không"))
                            self.lbl_weak.config(text="🔵 YẾU:\n" + ("\n".join(set(weak_texts)) if weak_texts else "Không"))
                            self.render_avatar(all_tight_ids, all_weak_ids)

                        except Exception as e: pass
                        
                        # Vẽ khung xương AI gốc đè lên hình ảnh
                        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                self.export_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                scale = min(640/w, 480/h)
                nw, nh = int(w * scale), int(h * scale)
                res = cv2.resize(image, (nw, nh))
                canvas_img = np.zeros((480, 640, 3), dtype=np.uint8)
                canvas_img[(480-nh)//2:(480-nh)//2+nh, (640-nw)//2:(640-nw)//2+nw] = res
                
                img_tk = ImageTk.PhotoImage(image=Image.fromarray(canvas_img))
                self.canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
                self.canvas.image = img_tk 
            
            self.window.after(15, self.process_frame)

    def on_closing(self):
        self.stop_camera()
        self.window.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = RehatrackApp(root, "Rehatrack - Hệ chuyên gia lâm sàng NASM")
    root.mainloop()