# 🚀 10-Step Quickstart Guide

Get up and running with the Fruit Freshness Detector in 10 minutes.

---

### Step 1: Open Terminal in Project Folder
Open PowerShell or Command Prompt in this folder.

### Step 2: Create Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Verify Camera Hardware
```powershell
python camera_test.py
```
*Verify that the camera window opens and the green inspection box is visible. Press **Q** to exit.*

### Step 5: Collect Fresh Fruit Samples
```powershell
python app.py collect --split train --label fresh --id apple001
```
*Place a fresh apple in the green square. Press **S** 5–10 times while rotating the fruit. Press **Q** when done.*

### Step 6: Collect Rotten Fruit Samples
```powershell
python app.py collect --split train --label rotten --id apple101
```
*Place a visibly deteriorated apple in the green square. Press **S** 5–10 times. Press **Q** when done.*

### Step 7: Collect Validation Samples (Independent Specimens!)
```powershell
python app.py collect --split val --label fresh --id apple002
python app.py collect --split val --label rotten --id apple102
```

### Step 8: Audit Dataset Hygiene
```powershell
python audit_dataset.py
```
*Ensure all checks PASS (directory structure, image readability, zero leakage).*

### Step 9: Train the AI Classifier
```powershell
python train.py --epochs 15 --batch-size 16
```
*This trains MobileNetV2, saves the best checkpoint, exports `artifacts/fruit.onnx`, and verifies ONNX accuracy against PyTorch.*

### Step 10: Run Live Freshness Scanner!
```powershell
python app.py scan --threshold 0.80 --alert-count 3
```
*Place fruit inside the square and press **S** to scan! Inspect results in `logs/scans.csv`.*
