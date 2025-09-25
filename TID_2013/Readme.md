# TID2013 Dataset – Processing and Modeling Report

---

## 1. Introduction
The **TID2013 dataset** is one of the most widely used benchmarks for full-reference Image Quality Assessment (IQA).  
It extends the earlier TID2008 dataset with additional distortions and subjective evaluations, making it valuable for training and testing perceptual quality models.  
This project adapts TID2013 for **patch-based perceptual quality prediction** in the context of pixel interpolation research.

---

## 2. Dataset Description
- **Content**:  
  - 25 reference images (24 natural + 1 synthetic).  
  - Each distorted by **24 distortion types × 5 levels** → 3,000 distorted images.  
- **Annotations**:  
  - **Mean Opinion Scores (MOS)** collected from ~985 subjective experiments.  
  - Data obtained from five countries (Finland, France, Italy, Ukraine, USA).  
- **Applications**: benchmarking IQA metrics (PSNR, SSIM, FSIMc) and training models for perceptual tasks such as compression, denoising, watermarking, and medical imaging. 
- **Reference Paper**: [Ponomarenko et al., 2013](https://www.ponomarenko.info/papers/tid2013.pdf) – *Image database TID2013: Peculiarities, results and perspectives*.  
- See the full wiki page of the dataset [here](https://github.com/HITProjects/PixelQuality/wiki/Mikhailuk-Dataset).
---

## 3. Patch Generation Process

### 3.1 Input
- Reference + distorted image pairs with MOS scores.  
- A subset of **1,000 distorted images** selected for processing.  

### 3.2 Patch Extraction
- Each image divided into **20×20 RGB patches**.  
- Central **12×12 region** treated as Region of Interest (RoI).  

### 3.3 Fusion Strategies
1. **Copy** – distorted ROI replaced with reference ROI.  
2. **Blend** – clean and distorted ROIs combined as weighted sum (α=0.5).  

### 3.4 Metadata
For each patch pair:
- Compute **PSNR** between clean and distorted.  
- Save metadata in JSON format:  
{ image_id, patch_id, coords, distortion_type, MOS, PSNR, fusion_type, alpha }


**Output**: A dataset of clean and fused patches with full metadata annotations, ready for model training.

---

## 4. Model Architecture
The **TinyPatchRegressor** CNN was implemented to predict perceptual quality from patch pairs.  

- **Input**: 6-channel tensor (clean RGB + distorted RGB, 20×20).  
- **Layers**:  
- Convolutional layers with GroupNorm and ReLU activations.  
- Global Average Pooling → Dropout.  
- Fully connected output layer → single regression score.  
- **Training Setup**:  
- Loss: Mean Squared Error (MSE).  
- Optimizer: Adam (lr=0.001).  
- Batch size = 16, Epochs = 15.  
- Early stopping (patience=5, min_delta=0.0005).  
- Mixed-precision training with GradScaler for efficiency.  

---

## 5. Experimental Results

### Synthetic Test Partition
- Converged in ~10–12 epochs with stable loss reduction.  
- **Final loss**: ~0.0154.  
- Captured subtle perceptual differences across multiple distortions.  
- Mixed-precision improved training speed and stability.  

### Samsung Test Partition
- Maintained good accuracy after **threshold calibration**.  
- Confusion matrices aligned with expected **Red / Orange / Green** quality bins.  
- Demonstrated solid generalization despite domain differences.  

---

## 6. Key Insights
- Fusion-based patch generation (copy + blend) provided strong supervision signals.  
- Even a lightweight CNN was sufficient to capture perceptual quality variations at the patch level.  
- GroupNorm improved stability given the small batch size (16).  

---

## 7. Future Work
- Extend models with **deeper CNNs or residual blocks** to handle complex distortions.  
- Expand training with additional IQA/JND datasets (e.g., PIPAL, KonJND-1k) for more patch diversity.  
- Explore **larger patch sizes** and additional fusion strategies.  
- Benchmark against advanced IQA metrics (FSIMc, LPIPS, DISTS).  

---

## 8. Conclusion
The TID2013 dataset, when repurposed into fused patch pairs, enables reliable patch-level perceptual quality prediction.  
Results confirm that small CNNs can learn meaningful quality representations, though further dataset expansion and deeper models are expected to improve robustness and accuracy across diverse real-world conditions.

