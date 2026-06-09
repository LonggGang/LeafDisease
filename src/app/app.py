import customtkinter as ctk
from PIL import Image
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import cv2
from google import genai  # Thư viện gọi Gemini AI
import os

# 🔴 DÁN API KEY CỦA BẠN VÀO ĐÂY
GEMINI_API_KEY = "AQ.Ab8RN6K9KXbcAfXYfE4gyeu9SYymx-Mytmed2JLPfmgVJB1t1g"

class AppPhatHienBenhLa(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Hệ Thống Phát Hiện Bệnh Lá Cây - YOLO x Gemini x Meme")
        self.geometry("950x850") # Tăng nhẹ chiều cao để chứa cả text phản hồi dài của Gemini

        # Khởi tạo model YOLOv12
        self.model = YOLO("best_od_yolov12s_plantdoc.pt") 
        
        # Khởi tạo Client Gemini AI
        try:
            self.ai_client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            print(f"Lỗi khởi tạo Gemini Client: {e}")

        # --- GIAO DIỆN (UI) ---
        self.title_label = ctk.CTkLabel(self, text="🌱 AI PHÁT HIỆN VÀ TƯ VẤN BỆNH LÁ CÂY", font=ctk.CTkFont(size=22, weight="bold"))
        self.title_label.pack(pady=15)

        # Khung chứa 2 ảnh (Gốc và Dự đoán)
        self.frame_images = ctk.CTkFrame(self)
        self.frame_images.pack(fill="both", expand=True, padx=20, pady=5)

        self.lbl_image_orig = ctk.CTkLabel(self.frame_images, text="Chưa có ảnh gốc", width=420, height=320, fg_color="gray20")
        self.lbl_image_orig.pack(side="left", padx=15, expand=True)

        self.lbl_image_pred = ctk.CTkLabel(self.frame_images, text="Chưa có kết quả phân tích", width=420, height=320, fg_color="gray20")
        self.lbl_image_pred.pack(side="right", padx=15, expand=True)

        # Nút bấm tải ảnh
        self.btn_upload = ctk.CTkButton(self, text="Tải Ảnh Lá Cây Lên", command=self.upload_and_predict, font=ctk.CTkFont(size=16))
        self.btn_upload.pack(pady=15)

        # KHUNG CHỨA THÔNG TIN BÊN DƯỚI (Bên trái hiện chữ tư vấn, Bên phải hiện Meme đánh giá)
        self.frame_bottom = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_bottom.pack(fill="x", padx=20, pady=10)

        # Khung chữ tư vấn từ Gemini (Bên trái)
        self.frame_details = ctk.CTkFrame(self.frame_bottom, fg_color="gray15")
        self.frame_details.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.lbl_status = ctk.CTkLabel(self.frame_details, text="Trạng thái: Sẵn sàng", font=ctk.CTkFont(size=14, weight="bold"), text_color="#3B8ED0")
        self.lbl_status.pack(anchor="w", padx=15, pady=5)

        self.txt_details = ctk.CTkTextbox(self.frame_details, height=180, font=ctk.CTkFont(size=13), activate_scrollbars=True)
        self.txt_details.pack(fill="x", padx=15, pady=5)
        self.txt_details.insert("0.0", "Thông tin tư vấn từ chuyên gia Gemini AI và đánh giá độ tin cậy sẽ hiển thị ở đây...")
        self.txt_details.configure(state="disabled")

        # Khung chứa ảnh Meme (Bên phải)
        self.frame_meme = ctk.CTkFrame(self.frame_bottom, fg_color="gray15", width=220, height=225)
        self.frame_meme.pack(side="right", fill="both", padx=(10, 0))
        self.frame_meme.pack_propagate(False) # Giữ cố định kích thước khung meme

        self.lbl_meme = ctk.CTkLabel(self.frame_meme, text="Meme phản hồi", fg_color="gray20")
        self.lbl_meme.pack(fill="both", expand=True, padx=5, pady=5)

    def hoi_gemini_ve_benh(self, ten_benh_tieng_anh):
        """Hàm gửi câu hỏi sang Gemini API, có bọc lỗi chống spam Quota 429"""
        prompt = f"""
        Bạn là một chuyên gia hàng đầu về bệnh học thực vật và nông nghiệp.
        Hệ thống AI vừa phát hiện ra một chiếc lá bị bệnh có tên tiếng Anh là: '{ten_benh_tieng_anh}'.
        
        Hãy viết một đoạn phản hồi ngắn gọn, súc tích bằng Tiếng Việt gồm 3 phần rõ ràng:
        1. Dịch tên bệnh sang Tiếng Việt chuẩn nông nghiệp.
        2. TẠI SAO BỊ (NGUYÊN NHÂN): Giải thích ngắn gọn tác nhân (vi khuẩn/nấm/môi trường) gây ra bệnh này.
        3. BIỆN PHÁP XỬ LÝ: Đưa ra 3-4 hành động thực tế, cụ thể để cứu cây và phòng ngừa lan rộng.
        """
        try:
            # Sử dụng model gemini-2.0-flash để đảm bảo tốc độ và độ ổn định quota tốt hơn
            response = self.ai_client.models.generate_content(
                model='gemini-3.1-flash-lite', 
                contents=prompt,
            )
            return response.text
        except Exception as e:
            err_msg = str(e)
            # Tự động bắt lỗi Quota 429 bắt người dùng đợi
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                return "⏳ [Thông báo giới hạn]: Bạn đang bấm gọi AI quá nhanh nên Google tạm khóa tần suất (Lỗi Quota 429).\nVui lòng ĐỢI KHOẢNG 10 GIÂY rồi tải lại ảnh để hệ thống nhận phản hồi từ Gemini nhé!"
            
            # Các lỗi sập mạng hoặc lỗi 503 khác
            return f"❌ Không thể kết nối với Gemini AI do lỗi hệ thống bên ngoài.\n[Chi tiết lỗi]: {err_msg}\n\n👉 Khuyến nghị chung: Ngắt bỏ lá bệnh tránh lây lan, hạn chế tưới nước ban đêm."

    def cap_nhat_meme_theo_conf(self, conf_score):
        """Hàm xử lý logic phân loại độ tự tin và hiển thị ảnh meme tương ứng"""
        if conf_score < 0.40:
            image_name = "i-dont-know-im-just-guessing-here.webp"
            danh_gia = f"🔴 ĐỘ TỰ TIN THẤP ({conf_score*100:.1f}%): AI đang đoán bừa thôi chứ không chắc chắn đâu!"
        elif conf_score <= 0.75:
            image_name = "hmmm-im-not-64779224b7.jpg"
            danh_gia = f"🟡 ĐỘ TỰ TIN TRUNG BÌNH ({conf_score*100:.1f}%): Nhìn cũng giống đấy nhưng vẫn hơi nghi ngờ..."
        else:
            image_name = "lying-yeah-trust-me.png"
            danh_gia = f"🟢 ĐỘ TỰ TIN CAO ({conf_score*100:.1f}%): Chuẩn bệnh luôn rồi, tin tưởng tuyệt đối nha bro!"

        # Hiển thị ảnh Meme lên giao diện nếu file tồn tại
        if os.path.exists(image_name):
            img_meme = Image.open(image_name)
            img_meme_resized = img_meme.resize((210, 215)) # Thay đổi nhẹ kích thước ảnh meme cho khớp với frame
            ctk_img_meme = ctk.CTkImage(light_image=img_meme_resized, dark_image=img_meme_resized, size=(210, 215))
            self.lbl_meme.configure(image=ctk_img_meme, text="")
        else:
            self.lbl_meme.configure(image=None, text=f"Thiếu file ảnh:\n{image_name}")

        return danh_gia

    def upload_and_predict(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")])
        
        if file_path:
            # 1. Hiển thị ảnh gốc
            img_orig = Image.open(file_path)
            img_orig_resized = img_orig.resize((420, 320))
            ctk_img_orig = ctk.CTkImage(light_image=img_orig_resized, dark_image=img_orig_resized, size=(420, 320))
            self.lbl_image_orig.configure(image=ctk_img_orig, text="")

            # 2. Chạy Model YOLOv12 dự đoán vẽ khung
            results = self.model(file_path)
            
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            img_pred = Image.fromarray(res_rgb)
            img_pred_resized = img_pred.resize((420, 320))
            ctk_img_pred = ctk.CTkImage(light_image=img_pred_resized, dark_image=img_pred_resized, size=(420, 320))
            self.lbl_image_pred.configure(image=ctk_img_pred, text="")

            # 3. Xử lý logic kép: Đổi Meme offline & Gọi Gemini online
            boxes = results[0].boxes
            self.txt_details.configure(state="normal")
            self.txt_details.delete("0.0", "end")

            if len(boxes) == 0:
                self.lbl_status.configure(text="Trạng thái: Không tìm thấy vết bệnh.")
                self.txt_details.insert("0.0", "Hệ thống không phát hiện ra dấu hiệu bệnh nào trên mẫu lá này.")
                self.lbl_meme.configure(image=None, text="Không có dữ liệu\nđể phản hồi")
            else:
                cls_id = int(boxes[0].cls[0])
                ten_benh_tieng_anh = self.model.names[cls_id]
                conf_value = float(boxes[0].conf[0])

                # Bước A: Xử lý hiển thị Meme ngay lập tức (Chạy offline, không tốn thời gian chờ mạng)
                chuoi_danh_gia_meme = self.cap_nhat_meme_theo_conf(conf_value)
                
                # Bước B: Thông báo trạng thái đang gọi API
                self.lbl_status.configure(text=f"Trạng thái: Đang kết nối với Chuyên gia Gemini AI để phân tích bệnh '{ten_benh_tieng_anh}'...")
                self.update() # Buộc giao diện cập nhật ngay lập tức ảnh meme và dòng trạng thái

                # Bước C: Gọi Gemini API lấy câu trả lời tư vấn sâu
                ket_qua_tu_gemini = self.hoi_gemini_ve_benh(ten_benh_tieng_anh)
                
                # Bước D: Tổng hợp dữ liệu in ra khung Text kết quả
                text_hien_thi = f"💬 ĐÁNH GIÁ ĐỘ TIN CẬY CỦA AI:\n{chuoi_danh_gia_meme}\n"
                text_hien_thi += "--------------------------------------------------\n"
                text_hien_thi += f"{ket_qua_tu_gemini}"
                
                self.lbl_status.configure(text="Trạng thái: Đã kết xuất báo cáo hoàn chỉnh!")
                self.txt_details.insert("0.0", text_hien_thi)
            
            self.txt_details.configure(state="disabled")

if __name__ == "__main__":
    app = AppPhatHienBenhLa()
    app.mainloop()