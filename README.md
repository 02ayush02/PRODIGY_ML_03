🐱🐶 Cat vs. Dog Image Classification using SVM

This project demonstrates a practical approach to image classification using Support Vector Machines (SVM). The goal is to accurately distinguish between images of cats and dogs using computer vision techniques like HOG feature extraction and dimensionality reduction with PCA.

📂 Dataset
- `train/train`: Contains labeled images of cats and dogs used for training.
- `test1/test1`: Contains unlabeled test images to be classified as either cat or dog.

📌 Features Used
Instead of raw pixels, this model uses extracted features:
- **HOG (Histogram of Oriented Gradients)**: Captures edge and texture patterns from grayscale images.
- **PCA (Principal Component Analysis)**: Reduces the high-dimensional HOG features to 500 principal components for efficiency.

🧹 Data Preprocessing & Feature Engineering
- All images resized to **128x128** for uniformity.
- Converted to **grayscale** for better HOG performance.
- **HOG features** extracted with:
  - 9 orientations
  - 8×8 pixel per cell
  - 2×2 cells per block
- **PCA** applied to reduce the feature space from ~8100 to 500 while preserving ~95% variance.

🧠 Model: Support Vector Machine (SVM)
- Used `sklearn.svm.SVC` with RBF kernel.
- Performed **GridSearchCV** to tune `C` and `gamma` hyperparameters.
- Trained on a balanced subset of 8000 samples.
- Split into 75% training and 25% validation.

📈 Results
- **Validation Accuracy**: ~71.65%
- Evaluated with `accuracy_score` and `classification_report`.
- Demonstrated decent performance on unseen data for a traditional ML pipeline.

🔍 Limitations
- No deep learning used — accuracy is limited compared to CNNs.
- Sensitive to lighting conditions and image quality.
- Misclassifications on ambiguous or low-quality images.

🚀 Future Improvements
- Upgrade to deep learning models (e.g., CNN with transfer learning).
- Use color-based features or augment dataset with variations.
- Incorporate more training data and apply data augmentation.
- Use cross-validation for more robust evaluation.

📁 Output
- A trained model saved as `svm_model_hog_pca.pkl` (includes model, scaler, and PCA transformer).
- A submission file `submission.csv` is generated with the following format:

