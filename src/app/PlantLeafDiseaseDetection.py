import os
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import customtkinter as ctk
from ultralytics import YOLO
import time
import threading
from groq import Groq

class PlantDiseaseDetectorApp(ctk.CTk):
    def __init__(self):
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("green")    
        super().__init__()

        # tu dien ngon ngu
        self.current_lang = "EN"
        self.translations = {
            "EN": {
                "window_title": "Plant Leaf Disease Detection System - YOLOv12s",
                "main_title": "🌱 AI LEAF DISEASE DETECTION AND DIAGNOSIS",
                "lbl_orig": "Original Image",
                "lbl_pred": "Detection Result",
                "btn_upload": "Upload Leaf Image",
                "btn_analyze": "Analyze Disease",
                "status_ready": "Status: Ready",
                "status_selected": "Status: Image selected. Ready to analyze.",
                "status_running": "Status: Running YOLO model...",
                "status_no_disease": "Status: No symptoms detected.",
                "status_consulting": "Status: Consulting AI for",
                "status_done": "Status: Analysis report complete!",
                "txt_placeholder": "Consultation results from AI and reliability assessments will appear here...",
                "txt_no_disease": "No signs of disease found on this leaf sample.",
                "txt_connecting": "⏳ CONNECTING TO AI FOR DETAILED DIAGNOSIS...",
                "txt_det_result": "📋 Detection Result (YOLOv12):",
                "txt_disease_name": "Disease Name",
                "txt_confidence": "System Confidence",
                "ai_prompt": """You are a top expert in plant pathology and agriculture. 
The AI system has detected a diseased leaf with the English name: '{disease}'. 
Write a short response in English (no markdown, plain text) consisting of 3 clear parts, written like a diagnostic report for farmers: 
1. DISEASE NAME: Briefly explain this disease. 
2. CAUSE: Briefly explain the agent (bacteria/fungus/environment) causing it. 
3. TREATMENT: Provide 3-4 practical, specific actions to save the plant and prevent spreading."""
            },
            "VI": {
                "window_title": "Hệ thống Phát hiện Bệnh trên Lá - YOLOv12s",
                "main_title": "🌱 AI PHÁT HIỆN VÀ CHẨN ĐOÁN BỆNH TRÊN LÁ",
                "lbl_orig": "Ảnh Gốc",
                "lbl_pred": "Kết quả Nhận diện",
                "btn_upload": "Tải Ảnh Lá Lên",
                "btn_analyze": "Phân Tích Bệnh",
                "status_ready": "Trạng thái: Sẵn sàng",
                "status_selected": "Trạng thái: Đã chọn ảnh. Sẵn sàng phân tích.",
                "status_running": "Trạng thái: Đang chạy mô hình YOLO...",
                "status_no_disease": "Trạng thái: Không phát hiện triệu chứng bệnh.",
                "status_consulting": "Trạng thái: Đang tham vấn AI về bệnh",
                "status_done": "Trạng thái: Đã hoàn tất báo cáo phân tích!",
                "txt_placeholder": "Kết quả tham vấn từ AI và đánh giá độ tin cậy sẽ xuất hiện ở đây...",
                "txt_no_disease": "Không tìm thấy dấu hiệu bệnh trên mẫu lá này.",
                "txt_connecting": "⏳ ĐANG KẾT NỐI VỚI AI ĐỂ CHẨN ĐOÁN CHI TIẾT...",
                "txt_det_result": "📋 Kết quả Nhận diện (YOLOv12):",
                "txt_disease_name": "Tên Bệnh (Tiếng Anh)",
                "txt_confidence": "Độ tin cậy",
                "ai_prompt": """Bạn là một chuyên gia hàng đầu về bệnh học thực vật và nông nghiệp.
Hệ thống AI vừa phát hiện ra một chiếc lá bị bệnh có tên tiếng Anh là: '{disease}'.
Hãy viết một đoạn phản hồi ngắn gọn bằng Tiếng Việt không có markdown gồm 3 phần rõ ràng, viết như một bài báo cáo chẩn đoán bệnh cho người nông dân bằng tiếng Việt, không có markdown hay ký tự đặc biệt nào, chỉ có văn bản thuần túy, với nội dung như sau:
1. TÊN BỆNH: Dịch sang tiếng Việt và giải thích ngắn gọn về bệnh này.
2. NGUYÊN NHÂN: Giải thích ngắn gọn tác nhân (vi khuẩn/nấm/môi trường) gây ra bệnh này.
3. BIỆN PHÁP XỬ LÝ: Đưa ra 3-4 hành động thực tế, cụ thể để cứu cây và phòng ngừa lan rộng."""
            }
        }

        self.title(self.translations[self.current_lang]["window_title"])
        self.geometry("950x850") 
        self.selected_file_path = None
        self.ctk_img_orig = None
        self.ctk_img_pred = None

        # load model yolo
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "best_od_yolov12s_plantdoc.pt")
        self.model = YOLO(model_path) 
        
        # khoi tao client groq ai
        try:
            self.ai_client = Groq(api_key="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")
        except Exception as e:
            print(f"AI Client Initialization Error: {e}")

        # nut chon ngon ngu
        self.frame_lang = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_lang.pack(anchor="ne", padx=20, pady=(10, 0))
        
        self.lang_selector = ctk.CTkSegmentedButton(self.frame_lang, values=["EN", "VI"], command=self.change_language)
        self.lang_selector.set("EN")
        self.lang_selector.pack()

        # layout giao dien
        self.title_label = ctk.CTkLabel(self, text=self.translations[self.current_lang]["main_title"], font=ctk.CTkFont(size=22, weight="bold"))
        self.title_label.pack(pady=5)

        # khung anh
        self.frame_images = ctk.CTkFrame(self, fg_color="#D2E7D6")
        self.frame_images.pack(fill="both", expand=True, padx=20, pady=5)

        self.lbl_image_orig = ctk.CTkLabel(
            self.frame_images, 
            text=self.translations[self.current_lang]["lbl_orig"], 
            width=420, height=320, 
            fg_color="#F3F8F4", text_color="#556B2F"
        )
        self.lbl_image_orig.pack(side="left", padx=15, expand=True)

        self.lbl_image_pred = ctk.CTkLabel(
            self.frame_images, 
            text=self.translations[self.current_lang]["lbl_pred"], 
            width=420, height=320, 
            fg_color="#F3F8F4", text_color="#556B2F"
        )
        self.lbl_image_pred.pack(side="right", padx=15, expand=True)

        # khung nut bam
        self.frame_buttons = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_buttons.pack(pady=15)

        self.btn_upload = ctk.CTkButton(self.frame_buttons, text=self.translations[self.current_lang]["btn_upload"], command=self.upload_image, font=ctk.CTkFont(size=16))
        self.btn_upload.pack(side="left", padx=10)

        self.btn_analyze = ctk.CTkButton(self.frame_buttons, text=self.translations[self.current_lang]["btn_analyze"], command=self.start_analysis, font=ctk.CTkFont(size=16), state="disabled", fg_color="#27ae60")
        self.btn_analyze.pack(side="left", padx=10)

        # khung thong tin ben duoi
        self.frame_bottom = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_bottom.pack(fill="x", padx=20, pady=10)

        self.frame_details = ctk.CTkFrame(self.frame_bottom, fg_color="#E8F5E9")
        self.frame_details.pack(fill="both", expand=True)

        self.lbl_status = ctk.CTkLabel(self.frame_details, text=self.translations[self.current_lang]["status_ready"], font=ctk.CTkFont(size=14, weight="bold"), text_color="#1B5E20")
        self.lbl_status.pack(anchor="w", padx=15, pady=5)

        self.txt_details = ctk.CTkTextbox(
            self.frame_details, 
            height=220, 
            font=ctk.CTkFont(family="Arial", size=15), 
            activate_scrollbars=True, 
            fg_color="#FFFFFF", 
            text_color="#2C3E50"
        )
        self.txt_details.pack(fill="x", padx=15, pady=5)
        self.txt_details.insert("0.0", self.translations[self.current_lang]["txt_placeholder"])
        self.txt_details.configure(state="disabled")

    def change_language(self, choice):
        """cap nhat ngon ngu tren giao dien"""
        self.current_lang = choice
        t = self.translations[self.current_lang]
        
        self.title(t["window_title"])
        self.title_label.configure(text=t["main_title"])
        
        # chi cap nhat chu neu chua co anh
        if self.ctk_img_orig is None:
            self.lbl_image_orig.configure(text=t["lbl_orig"])
        if self.ctk_img_pred is None:
            self.lbl_image_pred.configure(text=t["lbl_pred"])
            
        self.btn_upload.configure(text=t["btn_upload"])
        self.btn_analyze.configure(text=t["btn_analyze"])
        
        # reset text neu dang ranh
        if self.btn_analyze.cget("state") == "disabled" and self.selected_file_path is None:
            self.lbl_status.configure(text=t["status_ready"])
            self.txt_details.configure(state="normal")
            self.txt_details.delete("0.0", "end")
            self.txt_details.insert("0.0", t["txt_placeholder"])
            self.txt_details.configure(state="disabled")
        elif self.btn_analyze.cget("state") == "normal":
            self.lbl_status.configure(text=t["status_selected"])

    def ask_gemini_about_disease(self, disease_name_en):
        """goi api groq theo dung ngon ngu"""
        prompt = self.translations[self.current_lang]["ai_prompt"].format(disease=disease_name_en)
        
        try:
            chat_completion = self.ai_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.3,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"❌ Connection Error / Lỗi kết nối"

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")])
        
        if file_path:
            self.selected_file_path = file_path
            
            self.lbl_image_orig.configure(image=None)
            self.lbl_image_orig._label.configure(image="")

            img_orig = Image.open(file_path)
            img_orig_resized = img_orig.resize((420, 320))
            self.ctk_img_orig = ctk.CTkImage(light_image=img_orig_resized, dark_image=img_orig_resized, size=(420, 320))
            self.lbl_image_orig.configure(image=self.ctk_img_orig, text="")
            
            self.lbl_image_pred.configure(image=None, text=self.translations[self.current_lang]["lbl_pred"])
            self.lbl_image_pred._label.configure(image="")

            self.btn_analyze.configure(state="normal")
            self.lbl_status.configure(text=self.translations[self.current_lang]["status_selected"])

    def start_analysis(self):
        if not self.selected_file_path:
            return

        self.btn_analyze.configure(state="disabled")
        self.btn_upload.configure(state="disabled")
        self.lbl_status.configure(text=self.translations[self.current_lang]["status_running"])

        analysis_thread = threading.Thread(target=self._async_analysis_worker, daemon=True)
        analysis_thread.start()

    def _async_analysis_worker(self):
        t = self.translations[self.current_lang]
        
        results = self.model(self.selected_file_path)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        img_pred = Image.fromarray(res_rgb)
        img_pred_resized = img_pred.resize((420, 320))
        
        def display_prediction():
            self.lbl_image_pred.configure(image=None)
            self.lbl_image_pred._label.configure(image="")
            self.ctk_img_pred = ctk.CTkImage(light_image=img_pred_resized, dark_image=img_pred_resized, size=(420, 320))
            self.lbl_image_pred.configure(image=self.ctk_img_pred, text="")
        self.after(0, display_prediction)

        boxes = results[0].boxes

        if len(boxes) == 0:
            def ui_no_symptoms():
                self.lbl_status.configure(text=t["status_no_disease"])
                self.txt_details.configure(state="normal")
                self.txt_details.delete("0.0", "end")
                self.txt_details.insert("0.0", t["txt_no_disease"])
                self.txt_details.configure(state="disabled")
                self.btn_analyze.configure(state="normal")
                self.btn_upload.configure(state="normal")
            self.after(0, ui_no_symptoms)
        else:
            cls_id = int(boxes[0].cls[0])
            disease_name_en = self.model.names[cls_id]
            confidence_score = float(boxes[0].conf[0]) * 100

            initial_info = (
                f"{t['txt_det_result']}\n"
                f"• {t['txt_disease_name']}: {disease_name_en}\n"
                f"• {t['txt_confidence']}: {confidence_score:.2f}%\n\n"
                f"--------------------------------------------------\n"
                f"{t['txt_connecting']}\n"
            )
            
            def ui_show_initial():
                self.lbl_status.configure(text=f"{t['status_consulting']} '{disease_name_en}'...")
                self.txt_details.configure(state="normal")
                self.txt_details.delete("0.0", "end")
                self.txt_details.insert("0.0", initial_info)
            self.after(0, ui_show_initial)

            gemini_result = self.ask_gemini_about_disease(disease_name_en)
            
            final_report = (
                f"{t['txt_det_result']}\n"
                f"• {t['txt_disease_name']}: {disease_name_en}\n"
                f"• {t['txt_confidence']}: {confidence_score:.2f}%\n\n"
                f"--------------------------------------------------\n"
                f"{gemini_result}"
            )
            
            def ui_show_final():
                self.txt_details.configure(state="normal")
                self.txt_details.delete("0.0", "end")
                self.txt_details.insert("0.0", final_report)
                self.txt_details.configure(state="disabled")
                self.lbl_status.configure(text=t["status_done"])
                self.btn_analyze.configure(state="normal")
                self.btn_upload.configure(state="normal")
            self.after(0, ui_show_final)

if __name__ == "__main__":
    app = PlantDiseaseDetectorApp()
    app.mainloop()