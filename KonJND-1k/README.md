# KonJND-1K Dataset and Model

## 1. Introduction
The KonJND-1K dataset is a successor to smaller Just-Noticeable Difference (JND) datasets and is notable for being the largest picture-wise JND (PJND) database to date. The dataset's purpose is to benchmark and train models for perceptual image compression by helping to determine the point at which compression artifacts become noticeable to the human eye.  

The project also includes a deep learning model trained to predict the PJND for new images.

---

## 2. Dataset Description

- **Content:**  
  The dataset contains 1,008 source images and their compressed versions using JPEG and BPG compression schemes.

- **Subjective Ratings:**  
  It includes over 61,030 PJND ratings collected from 503 unique crowd workers via Amazon Mechanical Turk (AMT), with an average of 42 ratings per image.

- **Methodology:**  
  The subjective PJND ratings were collected using a flicker test combined with a slider-based method, which was more efficient and perceptually sensitive than traditional binary search methods.

- **Reference Paper:**  
  "Large-scale crowdsourced subjective assessment of picturewise just noticeable difference" by Hanhe Lin, Vlad Hosu, Franz Hahn, and Dietmar Saupe.

---

## 3. Patch Generation Process
The dataset was used to create a set of synthetic image patches for training a deep learning model, where each patch contains a clean region with a small fused distorted center.

- **Regions of Interest:**  
  Patches are sampled from areas with high local variance (edges, textures, and detailed regions), as compression artifacts are more perceptible in these areas.

- **Balanced Score Sampling:**  
  Subjective ratings are normalized to a [0,1] scale and divided into 10 bins. Patches are selected to ensure **roughly equal representation** across all score bins, using the bin with the fewest samples as the target for each patch.

- **Patch Sampling:**  
  For each image, a patch location is randomly sampled with probabilities weighted by local variance. Up to 20 attempts are made to avoid flat or identical patches (MSE=0).

- **Fused Patches:**  
  Each clean 20×20 patch has its **central 12×12 region** replaced by the corresponding distorted patch region. This simulates localized distortions, teaching the model to detect subtle perceptual differences.

- **PSNR and Score Computation:**  
  The Mean Squared Error (MSE) between the clean and fused patches is computed, from which the PSNR is derived. The patch is also assigned a normalized subjective score from the ratings.

- **Saving Patches and Metadata:**  
  Both clean and fused patches are saved as PNG files. Metadata stored for each patch includes: unique ID, file names, PSNR, subjective score, original image ID, compression type, patch coordinates, and sampling method.

---

## 4. Model Architecture and Training

- **Architecture:**  
  The model is a Convolutional Neural Network (CNN) with four convolutional blocks followed by fully connected layers. It takes a 10-channel input:
  - 3 channels for the clean patch
  - 3 channels for the distorted patch
  - 3 channels for the absolute difference between clean and distorted
  - 1 channel for the local variance map of the distorted patch  

  The network outputs a single scalar score between 0 and 1 using a sigmoid activation.
  
- **Input Features:**  
  Each patch is processed into a stacked tensor with clean, distorted, difference, and variance channels. Data augmentation is applied on the training set:
  - Random horizontal and vertical flips
  - Random rotations
  - Color jitter (brightness, contrast, saturation)

- **Loss Function:**  
  A custom **CombinedLoss** is used:
  - Mean Squared Error (MSE) for regression accuracy
  - A "disparity term" that encourages high standard deviation in predictions, preventing the model from collapsing to a single score.

- **Training Setup:**  
  - Optimizer: Adam (learning rate = 1e-3)  
  - Batch size: 32  
  - Epochs: up to 50 with early stopping (patience = 5)
    
---

## 5. Experimental Results

- **Synthetic Test Partition:**  
  The model achieved an accuracy of 0.38.

- **Samsung Test Partition:**  
  The accuracy was 0.17, with the model tending to guess only one class. The confusion matrix shows a strong bias toward the "Medium" predicted label.

---

## 6. Insights and Next Steps
- The results indicate that an improved model is needed to better learn the differences between patches.

- The model also needs to avoid its tendency to guess the middle bin.

- Creating more patches may be necessary, as the current amount might not be sufficient for training.
