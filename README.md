# Intro to AI: Plant Leaf Disease Detection

A lightweight, enterprise-standard Deep Learning pipeline for detecting plant leaf diseases. 
Optimized for deployment on Edge/IoT devices, featuring a decoupled processing architecture designed to handle diverse agricultural datasets (like PlantDoc and IDADP) with extreme care for texture preservation and class balancing.

## 🚀 Quick Start (Linux / Kaggle / Cloud)

We have provided bash scripts to make running this project completely trivial on any Linux-based production environment.

### 1. Setup the Environment
Run the setup script. It will install PyTorch and create the necessary folder structures.
```bash
bash scripts/setup.sh
```

*Note: You must manually download your raw dataset (e.g., from Kaggle) and place it inside the `data/raw/` folder. Once placed, run `bash scripts/setup.sh` one more time. The script will automatically physically split the data into `train`, `val`, and `test` folders securely.*

### 2. Train the Model
Kick off the training loop. This script automatically reads your hyperparameters from the `configs/` folder.
```bash
bash scripts/train.sh
```
*(Advanced: You can override configs by running `bash scripts/train.sh path/to/custom_train.yaml`)*

### 3. Evaluate the Model
Once training finishes, evaluate your best checkpoint on the unseen test data.
```bash
bash scripts/evaluate.sh
```

---

## 💻 Manual Run (Windows / No Bash)
If you are on Windows and cannot run `.sh` scripts, you can run the Python commands directly:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Split the Dataset:**
   ```bash
   python src/utils/split_dataset.py --raw_dir data/raw --out_dir data
   ```
3. **Train:**
   ```bash
   python main.py --mode train --train_cfg configs/train.yaml --augment_cfg configs/augment.yaml
   ```
4. **Evaluate:**
   ```bash
   python main.py --mode eval --checkpoint checkpoints/best_model.pth
   ```