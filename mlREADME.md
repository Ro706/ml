## 🤖 Machine Learning Fundamentals Overview

This document summarizes core concepts in Machine Learning, covering the ML Pipeline, key Classification and Regression algorithms, model selection criteria, and an introduction to Neural Networks and Deep Learning.

### 1️⃣ UNIT 1: Machine Learning Pipeline ⚙️

The **Machine Learning Pipeline** is a structured, repeatable sequence of steps for solving an ML problem.

  * **Definition:** A structured sequence of steps for solving an ML problem, ensuring **repeatability**, **reduced errors**, and **modular design**.
  * **Stages:**
    1.  **Problem Definition:** Define the objective and identify the ML problem type (e.g., classification, regression).
    2.  **Data Collection:** Gather data from internal, public, or sensor sources.
    3.  **Data Pre-processing:** Handle **missing values** (imputation/deletion), **encode categorical features** (Label/One-Hot), perform **feature scaling** (Standardization/Min-Max), and detect **outliers** (Z-score/IQR).
    4.  **Dataset Splitting:** Separate data into **Training** (70–80%) and **Testing** (20–30%) sets.
    5.  **Model Training:** Fit the chosen algorithm (e.g., Decision Tree, SVM) on the training data.
    6.  **Model Evaluation:** Measure performance using metrics.
          * Classification: **Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC–AUC.**
          * Regression: **MAE, MSE, RMSE, R² Score.**
    7.  **Model Validation:** Fine-tune the model using **K-fold cross-validation** and **Hyperparameter tuning** (Grid/Random Search).

-----

### 2️⃣ UNIT 2: Classification Algorithms 🏷️

**Classification** is the task of predicting a category or label (e.g., Spam vs. Not Spam).

  * **Algorithms:**

      * **Logistic Regression:** Uses the **Sigmoid function** to output a probability between 0 and 1. Fast and interpretable.
      * **Decision Tree:** Node-based structure that splits on the best feature; prone to **overfitting** (solved by pruning).
      * **Random Forest:** An ensemble of many Decision Trees that vote for the majority class; reduces overfitting.
      * **K-Nearest Neighbors (KNN):** Distance-based, no training phase, sensitive to scaling.
      * **Support Vector Machine (SVM):** Separates classes using a **hyperplane**; uses kernels (linear, RBF).
      * **Naive Bayes:** Probabilistic, based on **Bayes theorem**, assumes feature independence.

  * **Evaluation:** Calculated metrics like **Precision** (correct positives out of predicted positives) and **Recall** (correct positives out of actual positives).

-----

### 3️⃣ UNIT 3: Regression Techniques 📉

**Regression** is the task of predicting continuous numeric values (e.g., House price, Temperature).

  * **Types:**

      * **Simple Linear Regression:** $y=mx+c$ (one independent variable).
      * **Multiple Linear Regression:** $y=b_0+b_1x_1+b_2x_2+...$ (multiple independent variables).
      * **Polynomial Regression:** Fits curves by adding powers of $X$.
      * **Ridge Regression:** Uses **L2 regularization** ($\text{Loss} = \text{RSS} + \lambda \sum w^2$) to prevent overfitting.
      * **Lasso Regression:** Uses **L1 regularization**; performs implicit **feature selection**.

  * **Evaluation Metrics:** **MAE, MSE** (Mean Squared Error), **RMSE, R² Score** ($R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$).

-----

### 4️⃣ UNIT 4: Algorithm Selection & Comparative Analysis ⚖️

Selecting the right algorithm is based on data and problem characteristics.

| Criterion | **Small Data** | **Large Data** | **Non-Linear Data** | **Text Data** |
| :--- | :--- | :--- | :--- | :--- |
| **Best Choice** | SVM | Random Forest / Log Reg | Random Forest, SVM (RBF) | Naive Bayes |

  * **Selection Criteria:**
      * **Data Size:** SVM for small; Random Forest/Logistic Regression for large.
      * **Feature Count:** SVM/Logistic Regression for many features.
      * **Linearity:** Linear Regression/Logistic Regression for linear data; Random Forest/SVM for non-linear data.
  * **Case Study Example:** Random Forest often chosen for high accuracy on mixed datasets (like Customer Churn prediction), while Logistic Regression is used for interpretability.

-----

### 5️⃣ UNIT 5: Neural Networks and Deep Learning 🧠

**Neural Networks (NNs)** are modeled after the human brain, consisting of interconnected **neurons** organized into **input, hidden, and output layers**.

  * **Key Concepts:**

      * **Activation Functions:** Control the output of a neuron (e.g., **Sigmoid, ReLU** ($\text{max}(0, x)$), **Tanh, Softmax**).
      * **Deep Learning:** Uses **multiple hidden layers** to learn complex feature hierarchies.
      * **Training:** Uses **Backpropagation** and **Gradient Descent**.

  * **Specialized Architectures:**

      * **Convolutional Neural Networks (CNNs):** Primarily used for **image data**; layers include Convolution and Pooling.

[Image of Convolutional Neural Network Architecture]

```
* **Recurrent Neural Networks (RNNs):** Used for **sequence data** (e.g., text, time series); variations like **LSTM** and **GRU** improve memory.
```

  * **Semi-Supervised Learning:** Addresses situations where only **part of the data is labeled** using techniques like **Self-training** and **Pseudo labeling**.
