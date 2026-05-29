HƯỚNG DẪN CODING AGENT: TÁI TẠO KIẾN TRÚC YOLO-LEAFNET TỪ YOLOv8

Mục tiêu (Objective): Tái tạo chính xác kiến trúc mạng YOLO-LeafNet được đề xuất trong bài báo "YOLO-LeafNet: a robust deep learning framework for multispecies plant disease detection".
Bản chất của mô hình: YOLO-LeafNet là một bản tùy biến dựa trên kiến trúc gốc của YOLOv8, tập trung vào việc sửa đổi phần "Backbone".
Thư viện yêu cầu: ultralytics (PyTorch).

1. Phân tích Kiến trúc Cần Sửa đổi (Architecture Modifications)

Dựa trên sơ đồ kiến trúc (Figure 5 trong bài báo), những thay đổi so với YOLOv8 gốc (cụ thể là YOLOv8s dựa trên số lượng tham số ~11.1M) chỉ diễn ra ở giai đoạn cuối của Backbone, ngay trước khi truyền đặc trưng sang khối Spatial Pyramid Pooling - Fast (SPPF).

Luồng dữ liệu gốc của YOLOv8 Backbone (Giai đoạn cuối):
... -> Lớp Conv (P5) -> Lớp C2f cuối cùng (Layer 8) -> Lớp SPPF (Layer 9) -> Neck...

Luồng dữ liệu của YOLO-LeafNet Backbone (Giai đoạn cuối):
... -> Lớp Conv (P5) -> Lớp C2f cuối cùng -> LỚP BATCH NORMALIZATION -> LỚP DROPOUT -> Lớp SPPF -> Neck...

2. Nhiệm vụ Coding cho Agent

Agent cần thực hiện 2 bước chính để tạo ra mô hình này trong framework Ultralytics.

Bước 2.1: Tạo File YAML Định nghĩa Cấu trúc Mạng Mới

Tạo một file định nghĩa kiến trúc mới (ví dụ: yolo_leafnet.yaml) bằng cách copy cấu trúc của yolov8.yaml và chèn thêm 2 layer mới vào cuối backbone.

Hành động của Agent: Tạo file models/yolo_leafnet.yaml với nội dung phần backbone như sau:

# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLO-LeafNet customized architecture

# Parameters
nc: 11  # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  s: [0.33, 0.50, 1024]  # Dựa trên số parameter ~11.1M trong bài báo, model scale là YOLOv8s

# YOLOv8.0n backbone (Modified for LeafNet)
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]  # 8: Lớp C2f cuối cùng
  
  # --- CÁC LAYER ĐƯỢC CHÈN THÊM VÀO ĐÂY CHO YOLO-LEAFNET ---
  - [-1, 1, BatchNorm2d, [1024]] # 9: Lớp Batch Normalization (tham số args phải khớp số channel đầu ra của C2f)
  - [-1, 1, Dropout, [0.5]]      # 10: Lớp Dropout (Giả định tỷ lệ dropout là 0.5 vì bài báo không ghi rõ p)
  # ---------------------------------------------------------
  
  - [-1, 1, SPPF, [1024, 5]]    # 11: Lớp SPPF (Index đẩy lên 11)

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 14

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 17 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 20 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]  # cat head P5 (CHÚ Ý: TRỎ LẠI INDEX 11 LÀ SPPF THAY VÌ 9 NHƯ BẢN GỐC)
  - [-1, 3, C2f, [1024]]  # 23 (P5/32-large)

  - [[17, 20, 23], 1, Detect, [nc]]  # Detect(P3, P4, P5)


Bước 2.2: Đăng ký module độc lập (Standalone Modules) trong Ultralytics Parser

Thư viện ultralytics mặc định có thể không có module BatchNorm2d và Dropout được đăng ký như một lớp độc lập (đứng riêng một dòng trong YAML) để hàm parse_model đọc.
Hành động của Agent: Cần sửa mã nguồn thư viện hoặc định nghĩa module trực tiếp trong mã chạy.

Phương án tối ưu (Không sửa mã thư viện): Agent tạo một file Python (ví dụ: train_leafnet.py) để khởi tạo model và nạp các module PyTorch chuẩn thông qua việc đăng ký (Monkey patching hàm parse của Ultralytics) hoặc tạo một custom block chứa 3 module: C2f + BN + Dropout.

Tuy nhiên, cách dễ nhất (Monkey Patch / Custom Block) cho Agent:
Yêu cầu Agent tạo một custom module gom 3 chức năng:

Trong script training, định nghĩa một class mới:

import torch.nn as nn
from ultralytics.nn.modules import C2f

class C2f_BNDropout(nn.Module):
    """Khối C2f đi kèm với Batch Normalization và Dropout áp dụng ở cuối mạng backbone."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, p=0.5):
        super().__init__()
        self.c2f = C2f(c1, c2, n, shortcut, g, e)
        self.bn = nn.BatchNorm2d(c2)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = self.c2f(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


Sau đó, thay vì dùng file YAML có 3 dòng riêng lẻ, Agent sửa file YAML mục Backbone thành:

  # ... [Các lớp Conv trước đó]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f_BNDropout, [1024, True, 0.5]] # 8: Khối tùy biến (c2f + BN + Dropout)
  - [-1, 1, SPPF, [1024, 5]]    # 9: Lớp SPPF (Index giữ nguyên là 9 như bản gốc)


(Lưu ý: Nếu dùng cách này, phần head không cần phải sửa index nối (Concat) vì độ dài backbone không thay đổi).

Yêu cầu Agent map C2f_BNDropout vào mảng các module hợp lệ trong hàm parse_model của module ultralytics/nn/tasks.py.

3. Script Khởi tạo Model (Kiểm tra Architecture)

Yêu cầu Agent viết script test kiến trúc trước khi huấn luyện để đảm bảo số lượng tham số đúng như dự kiến (~11 triệu tham số).

from ultralytics import YOLO

# 1. Khởi tạo mô hình từ file cấu hình YOLO-LeafNet mới tạo
model = YOLO('models/yolo_leafnet.yaml')

# 2. In thông tin tóm tắt cấu trúc mạng để kiểm tra xem lớp BN và Dropout đã nằm đúng vị trí ngay trước SPPF chưa.
model.info() 
# Agent phải đảm bảo output model.info() in ra các lớp ở cuối backbone tương tự thế này:
# ...
# 8                  -1  ... C2f...
# 9                  -1  ... BatchNorm2d...
# 10                 -1  ... Dropout...
# 11                 -1  ... SPPF...
