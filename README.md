🐱🐶 Cat vs Dog Image Classification using SVM with HOG + PCA
This project demonstrates a machine learning pipeline to classify images of cats and dogs using Support Vector Machine (SVM). We apply HOG feature extraction followed by PCA for dimensionality reduction to achieve robust performance.

📂 Dataset
Training Data: train/train/
Contains labeled images of cats and dogs (e.g., cat.1.jpg, dog.2.jpg).

Testing Data: test1/test1/
Contains unlabeled images (e.g., 1.jpg, 2.jpg) for which predictions are made.

🔧 How This Project Works
This project follows a systematic ML pipeline from feature extraction to classification:

HOG Feature Extraction:
Every image is resized to 128x128, converted to grayscale, and passed through the Histogram of Oriented Gradients (HOG) to capture edge features.

PCA Dimensionality Reduction:
We use PCA to reduce feature dimensions from ~8100 to 500 components while preserving most of the variance, improving training time and reducing overfitting.

Standardization:
Features are scaled using StandardScaler to normalize the data distribution before feeding into SVM.

Model Training with GridSearch:
The best SVM parameters (C, gamma) are chosen using GridSearchCV and the final model is trained on 8000 examples.

Prediction & Submission:
After preprocessing the test images, predictions are saved in a submission.csv file with filenames and predicted labels (cat or dog).

📌 Features Used
HOG (Histogram of Oriented Gradients):

orientations=9

pixels_per_cell=(8, 8)

cells_per_block=(2, 2)

🧹 Data Preprocessing
Converted 4-channel (RGBA) images to 3-channel (RGB) if necessary.

Resized all images to (128, 128) for consistency.

Converted images to grayscale before extracting HOG features.

Applied PCA to reduce dimensionality.

Scaled features using StandardScaler.

🧠 Model: Support Vector Machine (SVM)
Trained using sklearn.SVC with RBF kernel.

Parameters selected using GridSearchCV with 5-fold cross-validation:

C: [0.1, 1, 10]

gamma: ['scale', 0.1, 0.01]

Best Parameters Found:

C = 10, gamma = scale, kernel = rbf

📈 Results
Validation Accuracy: ~71.65%

Precision/Recall on Validation:

Cat: P = 0.71, R = 0.74

Dog: P = 0.73, R = 0.69

This accuracy is based on a split of 8000 images from the training set.

📁 Output
The model is saved as svm_model_hog_pca.pkl.

The prediction file is saved as submission.csv in the format:
filename,label
1.jpg,dog
2.jpg,cat
...

🛠️ Requirements
Make sure to install the required libraries:
pip install numpy pandas scikit-learn scikit-image matplotlib

🚀 Future Improvements
Try other models like Random Forest, CNN, or Gradient Boosting.

Use data augmentation to improve generalization.

Explore deeper features using CNN (e.g., with transfer learning).

Tune PCA components for better trade-off between speed and accuracy.
