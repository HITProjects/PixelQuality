### Samsung R&D Project: Perceived Quality of Pixel Interpolation (QPI)

My part of the project involved developing a **deep learning model** and a **regression model** to predict the perceived quality score of two pixel interpolation methods.

The models' purpose is to learn the just-noticeable difference (JND) score that a human subject would give to a pair of images - a clean image and a distorted version of it, from a given set of scores: {0.0, 0.5, 1.0}.

When evaluating the models, predicted scores were digitized to fit the ground truth scores. Digitization bounds were chosen such that the model has it's highest possible accuracy on the training set, then the same values were used to digitize test set predictions.

---

### Dataset and Exploratory Data Analysis (EDA)

The project utilized a **Samsung dataset** of 20x20 RGB images ("patches") in groups of 3: a clean image, a version given by a new interpolation method and a version given by a reference interpolation method.
The models were given pairs of images, each pair with a score in {0.0, 0.5, 1.0}.

* **PSNR Histogram**: A histogram of PSNR values per interpolation method showed that the new interpolation method has more noise on average.

  <img width="512" height="333" alt="image" src="https://github.com/user-attachments/assets/4c5dfb0c-030e-445c-b959-d9c6424c15de" />
* **Label Imbalance:** The dataset has a significant label imbalance, with 0.0 appearing the most (1,594 times), 1.0 about half as many times (867 times) and 0.5 about a quarter of the time (497 times).

---

### Model Architecture and Training

Two approachs were examined for this task: convolutional neural network (CNN) and regression models.

<ins> CNN </ins>
* **Architecture:** The model, has a total of 44,641 parameters. It consists of three sequential convolutional blocks, each with two convolutions. The channels expand from 6 to 16, then to 24, and finally to 48. There are also two max pooling layers to reduce the feature maps to 5x5. The final layer is a single fully connected layer that performs a regression task to map to a single output.

<img width="512" height="388" alt="image" src="https://github.com/user-attachments/assets/cbacc663-1f77-4384-ba01-62bdefd14c58" />

* **Data Augmentation:** Applied horizontal and vertical flips with a probability of 0.5 for each, and data normalized with the train set's mean and STD.
  
* **Loss Function and Optimizer:** Used a **Weighted SmoothL1 loss function** to address the label imbalance. The weights were calculated based on the inverse of the label counts, normalized. The model was trained using the **SGD optimizer** with a learning rate of 1e-2 and a momentum of 0.9.
  
* **Performance:** After 40 epochs, the training loss was 0.0214 and the test loss was 0.0242. After 60 epochs, these improved to a training loss of 0.0148 and a test loss of 0.0198. Overall, the model achieved a 0.816 accuracy score, with better performance on the classes 0.0 and 1.0, and a lesser preformance on 0.5.

<img width="495" height="512" alt="image" src="https://github.com/user-attachments/assets/d4a201fb-587c-40c8-86b6-934227fa27a1" />


<ins> Regression </ins>
* **Features**: Different hand-crafted features were tried (PSNR both original and HVS-M; SSIM; GMSD; ΔE2000; FSIM). Best preformance achieved used PSNR, SSIM and FSIM.

* **Architecture:** Different models were tested. Top performers were the ensembles GradientBoostingRegressor and RandomForestRegressor, as well as SVR (RBF kernel).

* **Performance:** Using the GradientBoostingRegressor, an accuracy of 0.815 was achieved.

<img width="495" height="512" alt="image" src="https://github.com/user-attachments/assets/d84f0849-990d-46ce-9fb2-bd1ad97b7fd8" />

  Similar to the CNN model, the regression models performed best on the classes 0.0 and 1.0 while strugling with 0.5.

<img width="495" height="512" alt="image" src="https://github.com/user-attachments/assets/20393d06-3a01-41a4-b776-242e626cce02" />

---

### Results and Next Steps

A **macro F1 score of 0.7535** and a **model accuracy of 0.8162** were achieved.

* **Confusion Matrix Analysis:** The model performed well in predicting scores of 0 and 1, but struggled with the 0.5 score. This is most likely due to the label imbalance in the dataset. The model's predictions were digitized into three bins based on bounds that maximize accuracy on the training set: 0-0.402, 0.402-0.648, and 0.648-1. The confusion matrix shows high precision and recall for scores 0 ("Red") and 1 ("Green"), but a lower score for 0.5 ("Orange").
* **Next Steps:** To improve preformance further, several approaches may be considered:
    * Explore more robust model architectures, such as those with residuals or inception modules.
    * Consider using synthetic data to augment the training set.
    * Separate the task into two stages: first, predict if the score is 0.5, and then, if not, predict if it is 0 or 1.
    * Reinforce the promising ensemble regression models (Gradient Boosting, Random Forest, SVM) with feature engineering, or incorporating a neural network model to learn the regression error.
