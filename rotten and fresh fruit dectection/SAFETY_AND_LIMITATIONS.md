# Safety Guidelines, Limitations & Ethical Disclaimers ⚠️

## 1. Food Safety & Regulatory Disclaimers

> [!CAUTION]
> **NOT A FOOD-SAFETY CERTIFICATION TOOL**:
> The Fruit Freshness Detector is an educational and engineering prototype designed for visual computer-vision demonstration.
> - It does **NOT** certify food safety.
> - It does **NOT** measure microbiological contamination (e.g., *Salmonella*, *Listeria*, *E. coli*, *Aspergillus*).
> - It does **NOT** detect dangerous mycotoxins (e.g., Patulin in bruised apples) that may penetrate deep into fruit flesh.
> - It does **NOT** guarantee that any fruit classified as "FRESH" is edible or safe for human consumption.
> - It does **NOT** calculate remaining shelf life in hours or days.

---

## 2. Interpretation of Machine Learning Output

### Model Confidence Score vs Freshness Percentage:
The numerical score produced by the classifier (e.g., `87.4%`) represents the **statistical certainty of the CNN's visual classification**, NOT a physical "percentage of freshness".

- **Correct Interpretation**: *"The model has an 87.4% confidence that this image visually matches the characteristics of fresh produce in its training data."*
- **Incorrect Interpretation**: *"This apple is 87.4% fresh and 12.6% rotten."*

### The Role of the `UNCERTAIN` State:
When a fruit's top prediction confidence falls below the configured threshold (default: 80%), the system reports `UNCERTAIN`.
- This reflects genuine epistemic ambiguity (e.g., unusual lighting, partial shadows, mild natural skin blemishes, or borderline visual state).
- The system deliberately avoids forced binary guessing on ambiguous samples.

---

## 3. Biological Specimen Handling & Laboratory Hygiene

When collecting data or conducting live demonstrations with deteriorated fruit samples:

1. **No Culturing of Dangerous Molds**:
   - Do **NOT** attempt to artificially cultivate dense mold or bacterial colonies in unsealed containers.
   - Use naturally aged specimens obtained from safe retail or domestic sources.
2. **Personal Protective Equipment (PPE)**:
   - Wear disposable nitrile gloves when handling spoiled, soft-rotted, or moldy produce.
   - Wash hands thoroughly with soap and water immediately following data collection sessions.
3. **Safe Disposal**:
   - Double-bag spoiled fruit specimens in biodegradable or trash bags before disposal.
   - Clean the inspection surface and camera mount with standard surface disinfectant after testing.

---

## 4. Technical System Limitations

### 1. 2D Single-Angle Field of View:
A single camera frame captures only the visible hemisphere facing the lens. Rot, punctures, or insect damage on the underside or backside of the fruit cannot be detected unless the operator rotates the fruit and performs an additional scan.

### 2. Illumination & Shadow Sensitivity:
Harsh specular glare, deep shadows, or extreme color temperatures (e.g., deep orange tungsten bulbs) can distort peel coloration and deceive the CNN. Always operate under consistent, diffuse indoor lighting.

### 3. Ripeness vs Rot Distinction:
Natural cosmetic ripening features (e.g., russeting on apples, sugar spots on ripe bananas) must be properly represented in the training dataset to prevent the model from misclassifying cosmetic ripening as decay.

### 4. Absence of Object Localization:
The baseline model is an image classifier, not a bounding-box object detector. If an empty table, human hand, or unsupported object is positioned inside the green inspection box, the model will output a classification score based on background textures. Operators must ensure exactly one fruit is placed inside the crop area.
