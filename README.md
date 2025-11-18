# **Machine Learning**

Machine Learning (ML) is a branch of Artificial Intelligence that enables computers to **learn patterns from data** and make predictions or decisions **without being explicitly programmed**.

---

## 🧠 **What is Machine Learning?**

Machine Learning focuses on discovering relationships in data so that a system can:

* **Predict outcomes**
* **Recognize patterns**
* **Make decisions automatically**

### **Key Concepts**

#### **ML Paradigms**

* **Supervised Learning** – Learn from labeled data (e.g., classification, regression)
* **Unsupervised Learning** – Find patterns in unlabeled data (e.g., clustering, dimensionality reduction)
* **Reinforcement Learning** – Learn to take actions that maximize reward

#### **Common ML Tasks**

* Classification
* Regression
* Clustering
* Recommendation
* Anomaly Detection

#### **Typical ML Workflow**

1. Collect & clean data
2. Split into training/validation/test sets
3. Select and train model
4. Evaluate metrics
5. Deploy & monitor

#### **Popular ML Libraries**

* Scikit-learn
* TensorFlow
* PyTorch
* XGBoost

> ⚠️ ML performance heavily depends on data quality, good features, correct model choice, and avoiding overfitting.

---

## ⚙️ **Environment Setup**

Create and manage a Conda environment:

```sh
conda create -n ml_env
conda activate ml_env
conda deactivate
```

Install required packages:

```sh
conda install --file requirements.txt
```

---

## ✂️ **What does `train_test_split(x, y, test_size=0.2)` mean?**

This function is used to **split your dataset** into:

* **Training set (80%)** – used to train the model
* **Testing set (20%)** – used to evaluate the model on unseen data

### **Parameter meanings**

* **`x`** → Input features (your dataset’s independent variables)
* **`y`** → Labels/targets (the values the model must learn to predict)
* **`test_size=0.2`** → 20% of the dataset is used for testing

### **Example**

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```

This returns:

* `x_train`, `y_train` → training data
* `x_test`, `y_test` → testing data

---

