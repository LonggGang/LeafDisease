import os
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import customtkinter as ctk
from ultralytics import YOLO
from google import genai 
import time
import threading
# 🔴 DÁN API KEY CỦA BẠN VÀO ĐÂY
#GEMINI_API_KEY = "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"
from groq import Groq

# Khởi tạo Client (Nhớ nạp GROQ_API_KEY vào môi trường hoặc dán trực tiếp)
class PlantDiseaseDetectorApp(ctk.CTk):
    def __init__(self):
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("green")    
        super().__init__()

        self.title("Plant Leaf Disease Detection System - YOLOv12s (Group 2 IntroAI)")
        self.geometry("950x850") 
        self.selected_file_path = None  # Biến lưu đường dẫn ảnh khi bấm nút Upload

        # Load YOLOv12 model
        self.model = YOLO("src/app/best_od_yolov12s_plantdoc.pt") 
        
        # Initialize Groq AI Client (retrieved from environment variable GROQ_API_KEY)
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("Warning: GROQ_API_KEY environment variable is not set.")
            
        try:
            self.ai_client = Groq(api_key=api_key) if api_key else None
        except Exception as e:
            print(f"Groq Client Initialization Error: {e}")
            self.ai_client = None

        # --- UI LAYOUT ---
        self.title_label = ctk.CTkLabel(self, text="🌱 AI LEAF DISEASE DETECTION AND DIAGNOSIS", font=ctk.CTkFont(size=22, weight="bold"))
        self.title_label.pack(pady=15)

        # Image Frames
        self.frame_images = ctk.CTkFrame(self, fg_color="#D2E7D6")
        self.frame_images.pack(fill="both", expand=True, padx=20, pady=5)

        self.lbl_image_orig = ctk.CTkLabel(
            self.frame_images, 
            text="Original Image", 
            width=420, 
            height=320, 
            fg_color="#F3F8F4",
            text_color="#556B2F"
        )
        self.lbl_image_orig.pack(side="left", padx=15, expand=True)

        self.lbl_image_pred = ctk.CTkLabel(
            self.frame_images, 
            text="Detection Result", 
            width=420, 
            height=320, 
            fg_color="#F3F8F4",
            text_color="#556B2F"
        )
        self.lbl_image_pred.pack(side="right", padx=15, expand=True)

        # --- KHU VỰC NÚT BẤM ---
        self.frame_buttons = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_buttons.pack(pady=15)

        # Nút 1: Tải ảnh lên
        self.btn_upload = ctk.CTkButton(self.frame_buttons, text="Upload Leaf Image", command=self.upload_image, font=ctk.CTkFont(size=16))
        self.btn_upload.pack(side="left", padx=10)

        # Nút 2: Phân tích bệnh (Mặc định bị mờ)
        self.btn_analyze = ctk.CTkButton(self.frame_buttons, text="Analyze Disease", command=self.start_analysis, font=ctk.CTkFont(size=16), state="disabled", fg_color="#27ae60")
        self.btn_analyze.pack(side="left", padx=10)

        # Bottom Info Frames
        self.frame_bottom = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_bottom.pack(fill="x", padx=20, pady=10)

        # Consultant Text Frame (Mở rộng hết cỡ vì đã bỏ Meme)
        self.frame_details = ctk.CTkFrame(self.frame_bottom, fg_color="#E8F5E9")
        self.frame_details.pack(fill="both", expand=True)

        self.lbl_status = ctk.CTkLabel(self.frame_details, text="Status: Ready", font=ctk.CTkFont(size=14, weight="bold"), text_color="#1B5E20")
        self.lbl_status.pack(anchor="w", padx=15, pady=5)

        # Hộp văn bản bên trong (Ép màu nền trắng #FFFFFF, chữ màu xám đậm #2C3E50 để dễ đọc)
        self.txt_details = ctk.CTkTextbox(self.frame_details, height=180, font=ctk.CTkFont(family="Arial", size=13), activate_scrollbars=True, fg_color="#FFFFFF", text_color="#2C3E50")
        self.txt_details.pack(fill="x", padx=15, pady=5)
        self.txt_details.insert("0.0", "Consultation results from Gemini AI and reliability assessments will appear here...")
        self.txt_details.configure(state="disabled")

    def ask_gemini_about_disease(self, disease_name_en):
        """Asynchronous execution bridge to fetch text from Groq API client"""
        if not self.ai_client:
            return "❌ Lỗi: Biến môi trường GROQ_API_KEY chưa được cấu hình. Vui lòng thiết lập biến môi trường này để sử dụng tính năng Chẩn đoán AI chi tiết."
            
        prompt = f"""
        Bạn là một chuyên gia hàng đầu về bệnh học thực vật và nông nghiệp.
        Hệ thống AI vừa phát hiện ra một chiếc lá bị bệnh có tên tiếng Anh là: '{disease_name_en}'.
        Hãy viết một đoạn phản hồi ngắn gọn bằng Tiếng Việt không có markdown gồm 3 phần rõ ràng, viết như một bài báo cáo chẩn đoán bệnh cho người nông dân bằng tiếng Việt, không có markdown hay ký tự đặc biệt nào, chỉ có văn bản thuần túy, với nội dung như sau:
        1. TÊN BỆNH: Dịch sang tiếng Việt và giải thích ngắn gọn về bệnh này.
        2. NGUYÊN NHÂN: Giải thích ngắn gọn tác nhân (vi khuẩn/nấm/môi trường) gây ra bệnh này.
        3. BIỆN PHÁP XỬ LÝ: Đưa ra 3-4 hành động thực tế, cụ thể để cứu cây và phòng ngừa lan rộng.
        """
        try:
            # Gọi API của Groq
            chat_completion = self.ai_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.1-8b-instant", # Hoặc "qwen/qwen3-32b"
                temperature=0.3, # Để AI trả lời nhất quán theo khung
            )
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            return f"❌ Lỗi kết nối: {e}"

    def upload_image(self):
        """Hàm xử lý khi người dùng chọn ảnh hành động 1"""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")])
        
        if file_path:
            self.selected_file_path = file_path
            
            # 1. Hiển thị ảnh gốc lên giao diện trước
            img_orig = Image.open(file_path)
            img_orig_resized = img_orig.resize((420, 320))
            ctk_img_orig = ctk.CTkImage(light_image=img_orig_resized, dark_image=img_orig_resized, size=(420, 320))
            self.lbl_image_orig.configure(image=ctk_img_orig, text="")
            
            # Xóa ảnh kết quả cũ nếu có
            self.lbl_image_pred.configure(image=None, text="Detection Result")

            # 2. Kích hoạt nút bấm phân tích ảnh
            self.btn_analyze.configure(state="normal")
            self.lbl_status.configure(text="Status: Image selected. Ready to analyze.")

    def start_analysis(self):
        if not self.selected_file_path:
            return

        # Khóa nút bấm lại tạm thời để người dùng không bấm liên tục khi đang xử lý
        self.btn_analyze.configure(state="disabled")
        self.btn_upload.configure(state="disabled")
        self.lbl_status.configure(text="Status: Running YOLO model...")

        # Tạo một luồng phụ chạy ngầm hàm xử lý nặng, không làm đơ Main Thread
        analysis_thread = threading.Thread(target=self._async_analysis_worker, daemon=True)
        analysis_thread.start()

    # 🌟 3. THÊM MỚI: Hàm công nhân ngầm xử lý toàn bộ logic nặng
    def _async_analysis_worker(self):
        # 1. Run YOLOv12 ngầm
        results = self.model(self.selected_file_path)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        img_pred = Image.fromarray(res_rgb)
        img_pred_resized = img_pred.resize((420, 320))
        ctk_img_pred = ctk.CTkImage(light_image=img_pred_resized, dark_image=img_pred_resized, size=(420, 320))
        
        # Đẩy ảnh kết quả bounding box lên giao diện luôn
        def update_pred_image_ui():
            ctk_img_pred = ctk.CTkImage(light_image=img_pred_resized, dark_image=img_pred_resized, size=(420, 320))
            self.lbl_image_pred.configure(image=ctk_img_pred, text="")
        
        self.after(0, update_pred_image_ui)
        #self.lbl_image_pred.configure(image=ctk_img_pred, text="")

        # 2. Logic xử lý văn bản theo từng giai đoạn
        boxes = results[0].boxes
        self.txt_details.configure(state="normal")
        self.txt_details.delete("0.0", "end")

        if len(boxes) == 0:
            self.lbl_status.configure(text="Status: No symptoms detected.")
            self.txt_details.insert("0.0", "No signs of disease found on this leaf sample.")
        else:
            cls_id = int(boxes[0].cls[0])
            disease_name_en = self.model.names[cls_id]
            
            # 🎯 BƯỚC ĐỔI: Lấy độ tự tin từ YOLO và quy đổi sang %
            confidence_score = float(boxes[0].conf[0]) * 100

            self.lbl_status.configure(text=f"Status: Consulting Gemini AI for '{disease_name_en}'...")

            # 🌟 GIAI ĐOẠN 1: Xuất hiện ngay lập tức thông tin YOLO + dòng chữ chờ
            initial_info = (
                f"📋 Detection Result (YOLOv12):\n"
                f"• Disease Name (English): {disease_name_en}\n"
                f"• System Confidence: {confidence_score:.2f}%\n\n"
                f"--------------------------------------------------\n"
                f"⏳ CONNECTING TO GEMINI AI FOR DETAILED DIAGNOSIS...\n"
            )
            self.txt_details.insert("0.0", initial_info)

            # Gọi API Gemini ngầm (App vẫn mượt, người dùng đang ngồi đọc thông tin bên trên)
            gemini_result = self.ask_gemini_about_disease(disease_name_en)
            
            # 🌟 GIAI ĐOẠN 2: Khi có kết quả Gemini, xóa chữ "Đang kết nối..." và chèn báo cáo chính thức vào
            self.txt_details.delete("0.0", "end")
            final_report = (
                f"📋 Detection Result (YOLOv12):\n"
                f"• Disease Name (English): {disease_name_en}\n"
                f"• System Confidence: {confidence_score:.2f}%\n\n"
                f"--------------------------------------------------\n"
                f"{gemini_result}"
            )
            self.txt_details.insert("0.0", final_report)
            self.lbl_status.configure(text="Status: Analysis report complete!")
        
        self.txt_details.configure(state="disabled")
        
        # Mở khóa lại các nút bấm
        self.btn_analyze.configure(state="normal")
        self.btn_upload.configure(state="normal")

if __name__ == "__main__":
    app = PlantDiseaseDetectorApp()
    app.mainloop()