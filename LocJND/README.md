# LocJND Dataset – Perceptual Quality Extension for PQPI

## Project Context

The **LocJND** dataset is part of the **Perceived Quality of Pixel Interpolation (PQPI)** project —  
an initiative to train deep learning models that **simulate the human perception of visual differences** in small image regions.

The project aims to extend Samsung’s **PQPI** dataset (which focuses on **pixel interpolation**)  
by introducing **LocJND**, a perceptual dataset derived from **compression-based distortions**.  
This extension bridges two domains — **compression artifacts** and **pixel interpolation quality** —  
to achieve a more general perceptual model.

---

## What LocJND Represents

- **3 reference images**, each with **100 compression levels**
- **43 participants**, providing **129 perceptual sensitivity points**
- Data collected via **human flicker tests** to determine perceptual thresholds
- **HEATMAPs** marking areas most visible to human observers
- Each sample includes:
  - Clean (reference) patch  
  - Compressed patch  
  - Corresponding heatmap  
  - Alpha-blended patch (`mixed`)

The **visibility label** for each patch is derived from the blending ratio:

\[
y = 1 - alpha]

| α | y (visibility) |
|---|----------------|
| 1 | 0 (invisible)  |
| 0 | 1 (fully visible) |

---

## 📂 Folder Structure

The folder hierarchy must remain **exactly as shown**,  
otherwise the notebooks will fail to locate paths and metadata files.

```python
PixelQuality/   
└── PythonProject/    
    ├── artifacts/                # Saved ONNX / TorchScript / model weights    
    ├── checkpoints/              # Training checkpoints (.ckpt)    
    ├── data/   
    │   ├── LocJND.json           # Metadata file   
    │   └── originals/    
    │       ├── clean/            # Reference images      
    │       ├── compressed/       # Distorted images    
    │       └── heatmaps/         # Visibility maps   
    │   
    ├── dataset/    
    │   ├── orig_patches/         # Extracted 20×20 reference patches   
    │   ├── mixed_patches/        # Blended patches     
    │   ├── blends_meta.csv       # Metadata of all blends    
    │   ├── blends_train.csv      # Train split   
    │   ├── blends_val.csv        # Validation split    
    │   ├── blends_test.csv       # Test split    
    │   └── patches_meta.csv      # Additional patch metadata   
    │   
    ├── notebooks/    
    │   ├── 01_EDA.ipynb    
    │   ├── 02_Train_Regression.ipynb   
    │   ├── 03_Train_Classification.ipynb   
    │   └── 04_Test_Samsung.ipynb   
    │   
    ├── .idea/    
    ├── .git/   
    └── README.md   
```

---



## Notebook Summaries

### 1️⃣ **01_EDA – Exploratory Data Analysis**

**Goals:**
- Validate dataset integrity and structure.
- Split into train / validation / test.
- Explore distributions and heatmap correlations.

**Key actions:**
- Loads `LocJND.json` and CSV metadata.
- Generates per-image and per-alpha visualizations.
- Ensures reproducible splitting at image level (`random_state=42`).

**Important insights:**
1. The **folder structure must stay fixed** — relative paths are hardcoded.
2. **Heatmap rings** are for **statistical visualization only** (not labels).
3. The **label (y)** is based on the **blending ratio (α)**,  
   not on spatial position in the heatmap.
4. The **heatmap** is used only to locate areas that are perceptually sensitive,  
   but **the actual label** is always computed as `y = 1 - α`.

---

### 2️⃣ **02_Train_Regression – Local Quality Prediction**

**Goal:**  
Train a CNN (PatchNetDual) to predict the perceptual visibility level per patch.

**Main steps:**
- Load `train/val/test` CSVs and datasets.
- Construct three patch inputs: `orig`, `mixed`, `delta`.
- Train a dual-branch CNN (PatchNetDual) using:
  - **Loss:** `MSE + λ_rank + λ_pearson`
  - **Optimizer:** Adam with cosine LR schedule
  - **Metrics:** RMSE, PLCC, MAE
  - **Techniques:** Gradient clipping, AMP, Early Stopping
- Save best model and export to ONNX / TorchScript.

**Results:**
```python
Train PLCC ≈ 0.90
Validation PLCC ≈ 0.84
Test PLCC ≈ 0.91
```

**Takeaway:**  
The regression model successfully learned to approximate human perception of local distortion intensity.

---

### 3️⃣ **03_Train_Classification – Discrete Quality Categorization**

**Goal:**  
Convert the regression task into classification with 3 discrete perceptual levels:
- **Green (Good)** → low visibility  
- **Orange (Medium)** → moderate visibility  
- **Red (Bad)** → high visibility

**Thresholding rule:**
```python
if alpha < 0.33: label = 0
elif alpha < 0.66: label = 1
else: label = 2
```
**Model:**  
CNN with CrossEntropyLoss   
Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix

**Results:**   
```python
Accuracy ≈ 0.41   
Green: 68%  |  Orange: 30% |  Red: 30%
```
**Insights:**

* Thresholds are a critical hyperparameter — small changes significantly impact accuracy.
* The middle (orange) class is ambiguous —
humans themselves struggle to define it consistently,
and the model reflects that uncertainty.
→ Consider binary classification (“Good / Bad”) for clearer separation.

---

### 4️⃣ 04_Test_Samsung – Cross-Domain Generalization

**Goal:**     
Evaluate the LocJND-trained model on Samsung’s PQPI dataset.

**Process:**

1. Load Samsung test_metadata.json
2. Map Samsung’s perceptual scores to LocJND-style class labels
3. Run evaluation loop to compute accuracy and confusion matrices

**Results:**
```python
LocJND test:  ~0.41
Samsung test: ~0.48
```
**Findings:**

* Reasonable generalization despite dataset differences.
* Class “Green” retained best accuracy.
* Confirms existence of domain gap between datasets.
* Suggests next phase: joint fine-tuning or ensemble learning.

--- 
## Key Research Insights

* LocJND enables data-driven perceptual learning based on compression visibility.
* The project demonstrates successful cross-domain adaptation:
a model trained on LocJND data generalized to Samsung PQPI tasks.
* The regression model achieved high correlation (PLCC ≈ 0.9),
validating the perceptual consistency of the dataset.

--- 

## Next Steps

1. Maintain current folder hierarchy to ensure all notebooks work out of the box.
2. Experiment with binary classification (2 classes) for sharper perceptual distinction.
3. Balance the dataset (oversampling or focal loss) to handle the middle category bias.
4. Perform multi-dataset fine-tuning (Samsung + LocJND) for domain adaptation.
5. Extend the regression model with deeper backbones or residual blocks.

--- 

## Citation & Attribution

This dataset and analysis were developed as part of   
the PixelQuality collaboration project (2025),    
focusing on perceptual modeling of pixel interpolation and compression visibility.    
**Lead for LocJND:** Yehonatan Ravoach    
**Samsung PQPI Lead:** Tomer Portal   

