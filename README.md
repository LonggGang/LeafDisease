# Hướng Dẫn Sử Dụng Hệ Thống Phát Hiện Và Chẩn Đoán Bệnh Lá Cây (YOLOv12s LeafNet)

Dự án này là hệ thống tích hợp phát hiện vật thể (Object Detection) dựa trên **YOLOv12s LeafNet** và phân loại bệnh (Classification) sử dụng mạng CNN tùy biến, kết hợp với Trợ lý AI (Llama 3.1 qua Groq API) để cung cấp báo cáo chi tiết về nguyên nhân và giải pháp xử lý bệnh cho người nông dân.

---

## 1. Cài Đặt Môi Trường (Setup Environment)

Hệ thống hỗ trợ chạy trên Windows và Linux. Thực hiện các bước sau để thiết lập môi trường:

### Bước 1: Tạo môi trường ảo (Virtual Environment)
Mở terminal tại thư mục gốc của dự án và chạy lệnh:
```powershell
# Trên Windows
python -m venv .venv
```

### Bước 2: Kích hoạt môi trường ảo
```powershell
# Trên Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Trên Windows (CMD)
.venv\Scripts\activate.bat

# Trên Linux/macOS
source .venv/bin/activate
```

### Bước 3: Cài đặt các thư viện cần thiết
Tất cả các thư viện cần thiết đã được khai báo đầy đủ trong tệp `requirements.txt`. Thực hiện cài đặt bằng lệnh:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Đường Dẫn Dữ Liệu (Dataset Configuration)

Để chạy huấn luyện và đánh giá, vui lòng tải các tập dữ liệu tương ứng về máy và điền đường dẫn tuyệt đối hoặc tương đối vào các cấu hình dưới đây:

1. **PlantDoc Dataset**:
   * Đường dẫn: [PlantDoc Dataset on Kaggle](https://www.kaggle.com/datasets/andresmgs/plantdec)
   * *Mô tả: Chứa thư mục `images/` và `labels/` cho ảnh gốc chưa crop.*

2. **PlantVillage Dataset**:
   * Đường dẫn: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
   * *Mô tả: Chứa các ảnh lá cây đã được crop theo các thư mục tương ứng với từng lớp bệnh.*

3. **PlantVillage for Object Detection (Nếu có)**:
   * Đường dẫn: [PlantVillage for Object Detection on Kaggle](https://www.kaggle.com/datasets/sebastianpalaciob/plantvillage-for-object-detection-yolo)

---

## 3. Kịch Bản Huấn Luyện & Đánh Giá (Scripts Train & Eval)

Mọi thao tác huấn luyện, đánh giá đều được thực hiện thông qua tệp cổng vào `main.py`.

### A. Mô hình Phát hiện bệnh (Detection Tasks - YOLOv12s LeafNet)

*   **Huấn luyện (Train)**:
    ```powershell
    python main.py --mode train --task detection --architecture yolo_leafnet --data "đường_dẫn_đến_PlantDoc_data.yaml" --epochs 50 --batch_size 16
    ```
*   **Đánh giá (Evaluation)**:
    ```powershell
    python main.py --mode eval --task detection --architecture yolo_leafnet --checkpoint "checkpoints/yolov12s_leafnet_best.pt" --data "đường_dẫn_đến_PlantDoc_data.yaml"
    ```
*   **Dự đoán ảnh đơn lẻ (Predict)**:
    ```powershell
    python main.py --mode predict --task detection --architecture yolo_leafnet --checkpoint "checkpoints/yolov12s_leafnet_best.pt" --image_path "đường_dẫn_ảnh.jpg"
    ```

### B. Mô hình Phân loại bệnh (Classification Tasks - Advanced CNN / V2PlantNet)

*   **Huấn luyện (Train)**:
    ```powershell
    python main.py --mode train --task classification --architecture advanced_cnn --data "đường_dẫn_thư_mục_PlantVillage" --epochs 20 --batch_size 32
    ```
*   **Đánh giá (Evaluation)**:
    ```powershell
    python main.py --mode eval --task classification --architecture advanced_cnn --checkpoint "checkpoints/best_classifier.pth" --data "đường_dẫn_thư_mục_PlantVillage"
    ```

---

## 4. Kịch Bản Chạy Ứng Dụng Giao Diện (GUI App)

Ứng dụng giao diện (GUI) được viết bằng thư viện `customtkinter` và tích hợp mô hình phát hiện **YOLOv12s LeafNet** cùng chẩn đoán AI bằng **Groq**.

### Bước 1: Thiết lập API Key cho Trợ lý AI (Groq)
Mô hình chẩn đoán chi tiết bệnh sử dụng Llama 3.1 thông qua Groq Cloud. Bạn cần đăng ký tài khoản miễn phí trên Groq để lấy API Key, sau đó nạp vào biến môi trường hệ thống.

*   **Thiết lập trên Windows (PowerShell)**:
    ```powershell
    $env:GROQ_API_KEY="gsk_your_actual_api_key_here"
    ```
*   **Thiết lập trên Windows (CMD)**:
    ```cmd
    set GROQ_API_KEY=gsk_your_actual_api_key_here
    ```
*   **Thiết lập trên Linux/macOS**:
    ```bash
    export GROQ_API_KEY="gsk_your_actual_api_key_here"
    ```

### Bước 2: Chạy ứng dụng
Mở môi trường ảo đã cài đặt đầy đủ thư viện và chạy tệp GUI:
```powershell
python src/app/PlantLeafDiseaseDetection.py
```

### Bước 3: Sử dụng ứng dụng
1. Bấm nút **"Upload Leaf Image"** để chọn ảnh lá cây bị bệnh từ máy tính của bạn.
2. Bấm nút **"Analyze Disease"** (nút này sẽ sáng lên sau khi tải ảnh).
3. Hệ thống sẽ:
   * Chạy mô hình **YOLOv12s LeafNet** cục bộ để vẽ hộp bao (Bounding Box) khoanh vùng bệnh trên lá.
   * Gửi thông tin bệnh phát hiện được lên **Groq AI** qua luồng ngầm (threading) để lấy chẩn đoán chi tiết bằng Tiếng Việt gồm: Tên bệnh, Nguyên nhân, Biện pháp xử lý thực tế mà không làm đơ hay giật giao diện.
