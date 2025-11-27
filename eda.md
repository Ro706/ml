# 📘 Exploratory Data Analysis (EDA) in Python – Step-by-Step Guide

This README provides a **complete, beginner-friendly, and industry-standard workflow** for performing **Exploratory Data Analysis (EDA)** in Python on **any dataset**.

---

# 📌 What is EDA?

**Exploratory Data Analysis (EDA)** is the process of **summarizing, visualizing, and understanding a dataset** before applying any machine learning model.
EDA helps identify:

* Missing values
* Outliers
* Data types
* Patterns and relationships
* Distribution of data
* Errors or inconsistencies

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn
```

---

# 🚀 How to Perform EDA in Python (Step-by-Step)

Below is the complete, explained workflow.

---

## 🧩 **1. Import Required Libraries**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

**Explanation:**

* `pandas` → loading & manipulating tabular data
* `numpy` → numerical operations
* `matplotlib`, `seaborn` → plotting & visualization

---

## 📂 **2. Load the Dataset**

```python
df = pd.read_csv("data.csv")
# OR:
df = pd.read_excel("data.xlsx")
```

**Explanation:**
Loads data into a **DataFrame**, similar to an Excel sheet, where all analysis will take place.

---

## 👀 **3. Basic Data Inspection**

```python
df.shape          # Size (rows, columns)
df.head()         # First 5 rows
df.info()         # Data types & nulls
df.describe()     # Summary stats
```

**Explanation:**
This step helps understand the dataset's structure, types, and initial quality.

---

## 🕳 **4. Check Missing Values**

```python
df.isnull().sum()
```

### Visualizing Missing Values

```python
sns.heatmap(df.isnull(), cbar=False)
plt.show()
```

**Explanation:**
Missing values can cause inaccurate results or break models.
This step identifies where data is incomplete.

---

## 📑 **5. Check for Duplicate Rows**

```python
df.duplicated().sum()
df = df.drop_duplicates()
```

**Explanation:**
Duplicate rows distort averages, counts, and model training.

---

## 📊 **6. Univariate Analysis (One Variable at a Time)**

### Numerical Columns

```python
df['age'].hist()
plt.show()

sns.boxplot(df['age'])
plt.show()
```

### Categorical Columns

```python
sns.countplot(x='gender', data=df)
plt.show()
```

**Explanation:**
Helps understand distribution, central tendency, and outliers of each column.

---

## 🔗 **7. Bivariate Analysis (Two Variables Relationship)**

### Scatter Plot (Numeric vs Numeric)

```python
sns.scatterplot(x='age', y='salary', data=df)
plt.show()
```

### Correlation Heatmap

```python
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
```

### Categorical vs Numeric

```python
sns.boxplot(x='gender', y='salary', data=df)
plt.show()
```

**Explanation:**
Shows how variables interact or influence each other (correlation, trends).

---

## 🧠 **8. Multivariate Analysis (3+ Variables)**

### Pairplot

```python
sns.pairplot(df)
plt.show()
```

### Groupby Summary

```python
df.groupby('gender')['salary'].mean()
```

**Explanation:**
Shows complex interactions and patterns between multiple variables.

---

## 🚨 **9. Outlier Detection (IQR Method)**

```python
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['age'] < Q1 - 1.5*IQR) | (df['age'] > Q3 + 1.5*IQR)]
outliers
```

**Explanation:**
Outliers can distort statistical calculations and must be identified early.

---

## 📈 **10. Check Distribution Shape**

```python
sns.histplot(df['age'], kde=True)
plt.show()

sns.kdeplot(df['salary'])
plt.show()
```

**Explanation:**
Shows whether data is normal, skewed, or contains multiple peaks — important for modeling.

---

## 🧪 **11. Feature Engineering (Optional but Useful)**

```python
df['income_per_age'] = df['salary'] / df['age']
```

**Explanation:**
Creates new meaningful features using existing data, improving insights and model performance.

---

## 📝 **12. Final Summary of Findings**

After completing EDA, summarize:

* Missing values found
* Outliers detected
* Data imbalance
* Correlations
* Important patterns
* Data quality issues
* Insights for modeling

**Example Summary:**

> “Dataset contains 1000 rows and 12 columns. Age and Salary are right-skewed with multiple outliers. Gender distribution is imbalanced. Age and Experience have strong correlation (0.82). 5% values missing in Income column which needs imputation.”

---

# 🎯 Full EDA Code (Copy-Paste Ready)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data.csv")

# Basic info
print(df.shape)
print(df.info())
print(df.describe())

# Missing values
print(df.isnull().sum())
sns.heatmap(df.isnull(), cbar=False)
plt.show()

# Duplicates
print(df.duplicated().sum())
df = df.drop_duplicates()

# Univariate Analysis
df.hist(figsize=(12,8))
plt.show()

# Categorical Countplots
for col in df.select_dtypes(include='object').columns:
    sns.countplot(x=col, data=df)
    plt.show()

# Bivariate Analysis
sns.pairplot(df)
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Outliers (IQR)
for col in df.select_dtypes(include=np.number).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(col, "outliers:", len(outliers))
```



If you want, I can also generate a **Jupyter Notebook (.ipynb)** version of this EDA or tailor it for a **specific dataset**.
