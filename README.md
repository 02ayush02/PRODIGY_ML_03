# 🐱🐶 Cats vs Dogs Image Classification using SVM

This project demonstrates an image classification pipeline to distinguish between cats and dogs using **Support Vector Machine (SVM)**. It uses **HOG (Histogram of Oriented Gradients)** features along with **PCA** and **Standardization** to train an effective classifier on labeled image data.

---

## 📂 Dataset

- `train/` — Folder containing labeled images (e.g., `cat.0.jpg`, `dog.1.jpg`) used for training.
- `test1/` — Folder containing unlabeled images for prediction.
- `svm_model_hog_pca.pkl` — The saved trained model including the SVM, scaler, and PCA.

---

## ⚙️ How the Project Works

### 1. **Training Phase** (`Dogs_vs_Cats.ipynb`)
- Load and preprocess images from `train/`
- Extract HOG features, apply PCA, and scale the features
- Train an SVM classifier with hyperparameter tuning using GridSearchCV
- Evaluate the model on a validation set
- Save the trained model (`svm_model_hog_pca.pkl`)

### 2. **Prediction Phase** (`Dogs-vs-Cats-Test.ipynb`)
- Load the saved model, scaler, and PCA
- Preprocess and extract features from test images (`test1/`)
- Predict class labels (Cat or Dog)
- Save the predictions to `submission.csv`

---

## 📌 Features Used

- **HOG (Histogram of Oriented Gradients)**: Captures edge and shape information from grayscale images.
- **PCA (Principal Component Analysis)**: Reduces feature dimensionality to improve performance and generalization.
- **StandardScaler**: Standardizes features by removing the mean and scaling to unit variance.

---

## 🧹 Data Preprocessing

- All images are resized to **128x128** pixels.
- Converted to grayscale for consistency.
- Extracted HOG descriptors are reduced to 500 dimensions using PCA.
- Features are standardized using `StandardScaler`.

---

## 🧠 Model: SVM (Support Vector Machine)

- SVM with RBF kernel (`sklearn.svm.SVC`) is used.
- Hyperparameters (`C`, `gamma`) tuned using `GridSearchCV` with 5-fold cross-validation.
- The model is evaluated on a 75:25 train-validation split.

---

## 📈 Results

- Achieved accuracy of ~**71.35%** on the validation set *(replace with actual value)*.
- Predictions are exported in the following format:

```csv
filename,label
1.jpg,dog
2.jpg,cat
...
