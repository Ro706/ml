# Introduction to Machine Learning

Machine Learning (ML) is a subfield of Artificial Intelligence that enables systems to learn patterns from data and make decisions without being explicitly programmed for every situation. Instead of writing rules manually, ML models identify relationships between inputs and outputs and use those patterns to perform tasks such as classification, prediction, clustering, and recommendation. <br>
ML works by feeding large amounts of data into algorithms, which adjust internal parameters to minimize errors. Over time, the model becomes better at mapping inputs to outputs. A typical ML workflow includes collecting data, cleaning it, selecting features, training a model, evaluating performance, and deploying it for real-world use.<br>
Machine Learning is widely used in applications like spam detection, face recognition, recommendation systems, fraud detection, speech recognition, self-driving cars, and medical diagnosis. The effectiveness of ML depends on the quality of data, algorithm choice, feature representation, and proper training methods. ML models also need continuous monitoring because real-world data distributions often change over time, leading to model drift.<br>

---

# Different Types of Learning

Machine Learning has three main categories based on how the model learns from data:

(a) Supervised Learning

The model learns from labeled data—each input has a known output. Example: predicting house prices, detecting spam emails. Algorithms include Linear Regression, SVM, Decision Trees, Random Forest, etc.

(b) Unsupervised Learning

The model learns hidden patterns from unlabeled data. It groups similar points or discovers structure. Examples: customer segmentation, anomaly detection. Algorithms include K-Means, PCA, Hierarchical Clustering, Apriori, etc.

(c) Reinforcement Learning

The model learns through trial and error, receiving rewards for good actions and penalties for bad ones. Used in robotics, games (like AlphaGo), and self-driving systems.

Each learning type suits different problem scenarios. Supervised is best when labels exist, unsupervised when discovering patterns, and reinforcement learning when interacting with an environment. <br>

---

# Preparing an ML Model

This step involves all activities before the actual model training. It includes:

 - Understanding the problem (classification, regression, clustering).

 - Collecting data from databases, sensors, surveys, or web scraping.

 - Cleaning data by removing errors, duplicates, missing values, and noise.

 - Splitting data into training, validation, and test sets.

 - Choosing an algorithm based on the problem type and dataset size.

 - Feature engineering, scaling, and dimensionality reduction.

A well-prepared dataset leads to a better-performing model, while poor preparation results in inaccurate predictions.

---

# Data Pre-processing
Data pre-processing is essential because raw data usually contains noise, missing values, or inconsistent formats. It improves model accuracy and training efficiency.

Common techniques include:

 - Handling missing values: mean/median imputation, forward fill, deleting rows.

 - Encoding categorical variables: One-Hot Encoding, Label Encoding.

 - Feature scaling:

  - Standardization: makes mean = 0, variance = 1.

  - Normalization: scales values between 0 and 1.

 - Outlier detection: using IQR or z-score.

 - Data transformation: log transform, Box-Cox transform.

 - Balancing imbalanced data: SMOTE, undersampling, oversampling.

Pre-processing drastically impacts learning—unprocessed data often produces
