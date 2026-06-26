========================================================================
HE THONG PHAT HIEN VA CHAN DOAN BENH LA CAY (YOLOv12s LeafNet)
========================================================================

Du an nay tich hop phat hien vat the (YOLOv12s LeafNet), phan loai benh (CNN), va Tro ly AI (Llama 3.1 qua Groq) de phan tich bao cao benh tren la.

1. CAI DAT MOI TRUONG
- Tao venv: python -m venv .venv
- Kich hoat venv (PowerShell): .venv\Scripts\Activate.ps1
- Cai dat thu vien: pip install -r requirements.txt

2. CHAY GIAO DIEN (GUI APP)
- Chay lenh sau de mo giao dien:
  python src/app/PlantLeafDiseaseDetection.py
- Nen cai dat lai Groq API key trong code truoc khi chay de khong bi loi.

3. HUONG DAN SU DUNG
- Bam "Upload Leaf Image" de chon anh la cay bi benh.
- Bam "Analyze Disease" de chan doan benh.
- He thong se dung model YOLOv12s de khoanh vung benh va gui ten benh len Groq AI de lay giai phap dieu tri chi tiet bang tieng Viet/tieng Anh.

4. CAC LENH KHAC (Hoc may)
- Train yolo: python main.py --mode train --task detection --architecture yolo_leafnet --data data.yaml
- Eval yolo: python main.py --mode eval --task detection --architecture yolo_leafnet --checkpoint best.pt --data data.yaml
- Train classifier: python main.py --mode train --task classification --architecture advanced_cnn --data data_dir
