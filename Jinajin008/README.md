# Jianjin008 Dataset – Dataset Report

---

## 1. Introduction
The **Jianjin008 dataset** is a large-scale Just Noticeable Difference (JND) dataset, originally derived from TID2008, TID2013, and KADID-10k Image Quality Assessment datasets.  
It was constructed to study **perceptual thresholds of human vision** under multiple distortion types, using controlled crowdsourced experiments.  
This project focuses on adapting Jianjin008 for **patch-based quality prediction** in the context of pixel interpolation research.

---

## 2. Dataset Description
- **Source**: TID2008, TID2013, KADID-10k.  
- **Construction**: crowdsourced **flicker test** via Amazon Mechanical Turk (AMT).  
  - ~30 raters per group, calibrated with screen scaling and fixed viewing distance.  
  - Three categorical outcomes: *No Flicker, Slight Flicker, Obvious Flicker*.  
- **Content**: 1,642 JND maps across 25 distortion types.  
- **Applications**: compression, enhancement, watermarking, transmission optimization, and perceptual image quality modeling.  
- **Reference Paper**: [Liu et al., 2023](https://arxiv.org/pdf/2303.02562) – *The First Comprehensive Dataset with Multiple Distortion Types for Visual JNDs*.  
- See the full wiki page of the dataset [here](https://github.com/HITProjects/PixelQuality/wiki/jianjin008-Dataset). 
---

## 3. Patch Generation Process
The dataset was repurposed into **patch pairs** (clean + distorted) for training deep models to predict perceptual quality.

### 3.1 Filtering
- Images with more than three distortion levels were excluded.  
- Grouping by `(image, distortion)`.  
- Bucketing into categories:  
  - **no_flicker** (stable)  
  - **same_score + blend** (equal MOS with synthetic fusion)  
  - **slight_flicker** (just noticeable differences)  
- Only the *slight_flicker* bucket was used for training data.

### 3.2 Patch Extraction
- Each image divided into **20×20 RGB patches**.  
- Central **12×12 region** marked as the Region of Interest (RoI).  

### 3.3 Fusion Strategies
1. **Copy** – distorted RoI replaced with clean reference.  
2. **Blend** – clean and distorted RoIs combined as a weighted sum (α=0.5).  

### 3.4 Metadata
For each patch pair:
- Compute **PSNR** between clean and distorted.  
- Store structured metadata in JSON, including:  
{ image_id, patch_id, coords, distortion_type, MOS, PSNR, fusion_type, alpha }


**Output**: A dataset of clean and fused patches with complete annotations, ready for model training and evaluation.

---

## 4. Model Architecture
The **TinyPatchRegressor** CNN was implemented for patch-level quality prediction.  

- **Input**: 9-channel tensor  
- clean RGB + distorted RGB + absolute difference (3 + 3 + 3).  
- **Layers**:  
- Convolutional layers with GroupNorm and ReLU activation.  
- Global Average Pooling.  
- Fully connected output layer → single regression score.  
- **Training Setup**:  
- Loss: SmoothL1 (β=0.05).  
- Optimizer: Adam (lr=0.002).  
- Batch size = 16, Epochs = 20.  
- Early stopping (patience=10, min_delta=1e-5).  

---

## 5. Experimental Results

### Synthetic Test Partition
- Model converged over 20 epochs with steadily decreasing MSE loss.  
- Training loss: **0.125**  
- Validation loss: **0.131**  
- Demonstrated ability to capture MOS-based quality variations.

### Samsung Test Partition
- Threshold calibration improved binning into **Green / Orange / Red**.  
- Accuracy and balanced accuracy were lower than synthetic results due to **domain gap**.  
- Frequent misclassification of *Orange* vs *Red* patches.  

---

## 6. Key Insights
- Blended patches provide smoother perceptual scale for the model.  
- Stable convergence on synthetic data shows reliable learning.  
- Main limitation: domain transfer to real datasets reduces classification accuracy.  

---

## 7. Future Work
- Employ deeper models (Residuals, Inception modules) or pretrained features.  
- Expand dataset with additional JND/IQA sources for greater diversity.  
- Enrich patch generation with varied blending strategies and distortion types.  
- Apply balanced training strategies to reduce misclassification of intermediate quality levels.  

---

## 8. Conclusion
The Jianjin008 dataset provides a strong foundation for studying perceptual quality at the patch level.  
By repurposing it into fused patch pairs with metadata, and training lightweight CNN models, it is possible to approximate human judgments of subtle visual differences.  
While synthetic test results are promising, bridging the domain gap to real-world data remains the primary challenge.

---
