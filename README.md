# 🎯 Support Vector Machines (SVM) – Binary Classification on Breast Cancer Dataset

## 📌 Objective

This project is part of my **AI & ML Internship – Task 7**.  
The goal is to **implement Support Vector Machines for binary classification** to predict whether a tumor is **Malignant (0)** or **Benign (1)** using the **Breast Cancer Wisconsin dataset**. Focus on linear and RBF kernels, decision boundary visualization, hyperparameter tuning, and cross-validation.

---

## 🛠 Tools & Libraries Used

|Tool / Library|Purpose|
|---|---|
|**Python**|Core programming language|
|**Pandas**|Data loading and wrangling (though minimal here)|
|**NumPy**|Numerical computations, array operations|
|**Matplotlib/Seaborn**|Data visualization and plotting decision boundaries|
|**Scikit-learn**|Dataset loading, preprocessing, SVM modeling, evaluation, and tuning|
|**Joblib**|Model serialization|
|**Google Colab**|Notebook execution environment|

---

## 🔄 Workflow – Step-by-Step Logic Flow

Text-based flowchart showing entire process:

[Start]  
↓  
Load dataset (`datasets.load_breast_cancer()`)  
↓  
Initial inspection → shape, features, target distribution  
↓  
Split into train/test (80/20) with `train_test_split`  
↓  
Standardize features with `StandardScaler`  
↓  
Reduce to 2D for visualization using `PCA(n_components=2)`  
↓  
Train Linear SVM (`SVC(kernel='linear')`) and RBF SVM (`SVC(kernel='rbf')`)  
↓  
Predict on test set and evaluate (accuracy, classification report)  
↓  
Visualize 2D decision boundaries with meshgrid plotting  
↓  
Tune hyperparameters (C, gamma) via `GridSearchCV` with 5-fold CV  
↓  
Evaluate tuned model → confusion matrix, CV scores  
↓  
Interpret support vectors and save model/outputs with Joblib/CSV  
↓  
[End]

---

## 🧪 Steps Performed in Detail

## 1️⃣ **Data Loading**

- Dataset: **Breast Cancer Wisconsin (Diagnostic) Data Set**
    
- Loaded using: `cancer = datasets.load_breast_cancer(); X = cancer.data; y = cancer.target`
    

## 2️⃣ **Data Preparation**

- Split data: `train_test_split(X, y, test_size=0.2, random_state=42)`
    
- Scaled features: `StandardScaler().fit_transform(X_train)`
    
- Dimensionality reduction for visualization: `PCA(n_components=2).fit_transform(X_train_scaled)`
    

## 3️⃣ **Model Training**

- Trained Linear SVM: `SVC(kernel='linear').fit(X_train_scaled, y_train)`
    
- Trained RBF SVM: `SVC(kernel='rbf').fit(X_train_scaled, y_train)`
    
- Trained 2D versions for boundary plotting.
    

## 4️⃣ **Predictions & Evaluation**

- Predictions with `.predict(X_test_scaled)`.
    
- Metrics: Accuracy, classification report, confusion matrix.
    
- Cross-validation: `cross_val_score` for robustness.
    

## 5️⃣ **Visualization**

- Plotted decision boundaries using meshgrid and contourf for linear and RBF kernels on 2D PCA data.
    
- Added scatterplot of training points with class hues.
    

## 6️⃣ **Hyperparameter Tuning**

- Grid search: `GridSearchCV` with params {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]} and 5-fold CV.
    
- Selected best model and evaluated on test set.
    

## 7️⃣ **Interpretation**

- Counted support vectors: `len(best_svm.support_)`.
    
- Plotted confusion matrix heatmap.
    

## 8️⃣ **Model Saving**

- Saved model: `joblib.dump(best_svm, 'svm_model.pkl')`
    
- Exported predictions: `pd.DataFrame(...).to_csv('predictions.csv')`
    

---

## 📚 Vocabulary of Functions & Commands Used

|Command / Function|Purpose|
|---|---|
|`datasets.load_breast_cancer()`|Loads built-in breast cancer dataset|
|`train_test_split(X, y, test_size, random_state)`|Splits dataset into train and test sets|
|`StandardScaler()`|Scales features to mean=0, std=1|
|`.fit_transform(data)`|Fits scaler to data and transforms|
|`.transform(data)`|Transforms new data using fitted scaler|
|`PCA(n_components=2)`|Reduces dimensionality to 2D for visualization|
|`SVC(kernel='linear' or 'rbf')`|Defines SVM classifier with specified kernel|
|`.fit(X_train, y_train)`|Trains SVM model on training set|
|`.predict(X_test)`|Predicts labels for new data|
|`accuracy_score(y_true, y_pred)`|Computes prediction accuracy|
|`classification_report(y_true, y_pred)`|Generates precision, recall, f1-score|
|`confusion_matrix(y_true, y_pred)`|Creates confusion matrix|
|`cross_val_score(model, X, y, cv)`|Performs k-fold cross-validation|
|`GridSearchCV(estimator, param_grid, cv)`|Tunes hyperparameters with grid search and CV|
|`np.meshgrid(x, y)`|Creates grid for plotting decision boundaries|
|`plt.contourf(xx, yy, Z)`|Plots filled contours for boundaries|
|`sns.scatterplot(x, y, hue)`|Plots scattered points with class colors|
|`joblib.dump(object, filename)`|Saves Python object to file|
|`pd.DataFrame().to_csv(filename)`|Exports DataFrame to CSV|

---

## 📊 Key Insights

- SVM excels in high-dimensional spaces via margin maximization and the kernel trick, with RBF handling non-linear separability better than linear (achieved ~98% accuracy post-tuning).
    
- Hyperparameter tuning (C for regularization, gamma for RBF spread) is crucial to balance underfitting/overfitting; cross-validation ensures generalizability.
    
- Visualization of decision boundaries highlights how RBF creates curved separations, critical for non-linear data like cancer features.
    
- Support vectors (~100-200) define the model; fewer indicate efficient classification. SVM's robustness to outliers makes it ideal for medical datasets.
    

---

## ✍ Prepared By

📄 README prepared by **Perplexity AI**
