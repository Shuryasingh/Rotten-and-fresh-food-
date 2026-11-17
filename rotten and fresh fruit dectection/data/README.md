# Dataset Directory Structure

This directory holds dataset splits for training, validation, and testing.

## Folder Hierarchy

```
data/
├── train/
│   ├── fresh/
│   └── rotten/
├── val/
│   ├── fresh/
│   └── rotten/
└── test/
    ├── fresh/
    └── rotten/
```

## Image Naming Convention

Every image filename must include a **physical fruit identity** prefix before a double underscore (`__`):

```
<fruit_id>__<unique_id>.jpg
```

**Examples:**
- `apple001__001.jpg`
- `apple001__view_top.jpg`
- `apple002__side.jpg`
- `apple101__rot_spot.jpg`

## Physical Specimen Isolation (Zero Leakage Rule)

> [!IMPORTANT]
> A single physical fruit must reside in **exactly one split**.
> If `apple001` is placed in `train/`, no views or frames of `apple001` can appear in `val/` or `test/`.

Before training, run:
```bash
python audit_dataset.py
```
This script validates directory structure, image integrity, and ensures no physical fruit ID or duplicate image crosses split boundaries.
