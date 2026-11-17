# Dataset Collection & Hygiene Guide 📸🍎

A high-performing, robust computer vision classifier depends entirely on the **quality, independence, and hygiene** of its training and evaluation data.

---

## 1. The Cardinal Rule: Physical Specimen Isolation (Zero Data Leakage)

> [!IMPORTANT]
> **One Physical Specimen = Exactly One Dataset Split**
> All photographic views, angles, and rotations of a single physical fruit must belong to the **same split**.

### Correct Dataset Allocation:
```
data/
├── train/
│   ├── fresh/  ──► apple001__01.jpg, apple001__02.jpg, apple002__01.jpg
│   └── rotten/ ──► apple101__01.jpg, apple101__02.jpg, apple102__01.jpg
├── val/
│   ├── fresh/  ──► apple003__01.jpg, apple003__02.jpg  (New physical fruit!)
│   └── rotten/ ──► apple103__01.jpg, apple103__02.jpg  (New physical fruit!)
└── test/
    ├── fresh/  ──► apple004__01.jpg, apple004__02.jpg  (Unseen physical fruit!)
    └── rotten/ ──► apple104__01.jpg, apple104__02.jpg  (Unseen physical fruit!)
```

### ❌ What Causes Severe Data Leakage:
```
# LEAKAGE ERROR: The model simply memorizes the unique markings of apple001!
train/fresh/apple001__view01.jpg
test/fresh/apple001__view02.jpg  <--- WRONG!
```
If images of the same physical fruit appear in both `train/` and `test/`, the neural network will memorize individual peel markings, producing falsely inflated test scores that fail catastrophically in real-world deployment.

Our automated `audit_dataset.py` script actively scans filenames and SHA-256 hashes to catch and reject this error before training.

---

## 2. Filename Convention

All image filenames must follow this exact format:
```
<fruit_id>__<unique_token>.jpg
```
- **`<fruit_id>`**: Unique identifier for the individual physical fruit specimen (e.g. `apple001`, `apple-specimen-42`). Alphanumeric characters and hyphens only (`[A-Za-z0-9-]+`).
- **`__`**: Double underscore delimiter.
- **`<unique_token>`**: Unique view or hex hash (e.g. `view01`, `3a8f1b`, `rot_spot`).

The built-in collection tool automatically generates filenames matching this standard:
```powershell
python app.py collect --split train --label fresh --id apple001
```

---

## 3. Capturing Meaningful Views (Avoiding Near-Duplicates)

### ⚠️ The Burst-Photography Trap:
Holding down the capture key to take 50 identical frames in 3 seconds creates **redundant near-duplicate data**. The model learns very little from identical frames.

### Recommended Capture Technique:
For each physical fruit specimen:
1. Place fruit in the green inspection square.
2. Press **S** to capture view #1 (Top/Stem).
3. Rotate the fruit ~90° horizontally. Press **S** (Side view A).
4. Rotate another ~90°. Press **S** (Side view B).
5. Flip fruit upside down. Press **S** (Calyx/Bottom view).
6. Tilt at an angle showing any localized surface defect or blemish. Press **S**.
7. Aim for **4 to 8 diverse views** per physical specimen.

---

## 4. Background & Lighting Hygiene (Preventing Clever Hans Bias)

### The "Clever Hans" Problem:
If you photograph all **fresh** fruits on a clean white desk under bright daylight, and all **rotten** fruits on a brown tray under warm indoor lighting, the neural network will learn to classify the **desk background and lighting temperature** rather than the fruit's condition!

### Guidelines for Neutral Photography:
- **Same Surface**: Use the identical cutting board, neutral mat, or table surface for both fresh and rotten specimens.
- **Same Lighting**: Photograph both fresh and rotten fruits under consistent diffuse indoor lighting.
- **Consistent Distance**: Keep the fruit centered inside the square crop so it occupies 60–80% of the ROI.
- **No Hand Occlusion**: Keep hands and fingers out of the crop frame during image capture.

---

## 5. Target Dataset Size & Split Ratio

A sensible starting target for a school innovation project:
- **Training (70%)**: ~25–35 physical fruits per class (approx. 150–250 total images per class).
- **Validation (15%)**: ~5–8 independent physical fruits per class (approx. 30–50 images per class).
- **Test (15%)**: ~5–8 independent physical fruits per class (approx. 30–50 images per class).

*Note: Independence of physical specimens is far more critical than raw image count. 30 diverse physical fruits provide much better generalization than 1,000 burst shots of 2 fruits.*

---

## 6. Pre-Training Dataset Audit

Always verify your dataset hygiene before launching model training:
```powershell
python audit_dataset.py
```
Fix any reported format errors, corrupt files, or cross-split identity leaks before running `train.py`.
