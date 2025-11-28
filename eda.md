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

# 1. Load data
# Replace 'data.csv' with your actual file path
df = pd.read_csv("data.csv")

# 2. Basic info
print("--- Shape ---")
print(df.shape)
print("\n--- Info ---")
print(df.info())
print("\n--- Statistics ---")
print(df.describe())

# 3. Check Missing values
print("\n--- Missing Values Before Fix ---")
print(df.isnull().sum())
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Map")
plt.show()

# ---------------------------------------------------------
# 4. HANDLE MISSING VALUES (The New Section)
# ---------------------------------------------------------

# A. Numerical Data: Replace with Median
# We use Median because it is robust to outliers.
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled missing numeric values in '{col}' with Median: {median_val}")

# B. Categorical Data: Replace with Mode
# We use Mode (most frequent value) for text/categories.
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"Filled missing categorical values in '{col}' with Mode: {mode_val}")

# Verify clean-up
print("\n--- Missing Values After Fix ---")
print(df.isnull().sum().sum())
# ---------------------------------------------------------

# 5. Duplicates
print("\n--- Duplicates ---")
print(f"Duplicates found: {df.duplicated().sum()}")
df = df.drop_duplicates()
print("Duplicates dropped.")

# 6. Univariate Analysis
# Histograms for all numerical columns
df.hist(figsize=(12, 8), bins=20)
plt.suptitle("Univariate Analysis: Numerical Distributions")
plt.show()

# 7. Categorical Countplots
# Loops through all object columns to show frequency
for col in df.select_dtypes(include='object').columns:
    # Optional: check if too many categories exist to avoid messy plots
    if df[col].nunique() < 20: 
        plt.figure(figsize=(8, 4))
        sns.countplot(x=col, data=df)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45)
        plt.show()
    else:
        print(f"Skipping plot for {col}: too many unique values ({df[col].nunique()})")

# 8. Bivariate Analysis
# Pairplot (Scatter plots for all numerical pairs)
print("Generating Pairplot...")
sns.pairplot(df)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 9. Outliers (IQR Method)
print("\n--- Outlier Detection (IQR) ---")
for col in df.select_dtypes(include=np.number).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    if len(outliers) > 0:
        print(f"{col}: {len(outliers)} outliers found")
    else:
        print(f"{col}: No outliers")
```



If you want, I can also generate a **Jupyter Notebook (.ipynb)** version of this EDA or tailor it for a **specific dataset**.

## How to handle missing value from the given dataset

### 1. Numerical Data (Continuous or Discrete)

For numerical columns, you must examine the **distribution** of the existing, non-missing values, typically by viewing a histogram.

| Statistical Measure | When to Use It | Why Choose It | How to Code (Pandas) |
| :--- | :--- | :--- | :--- |
| **Median** | Data is **skewed** (not symmetrical) or contains **outliers** (e.g., income, house prices). | It's **robust** (unaffected by extreme values), ensuring the filled value doesn't distort the true center of the data. | `df[col].fillna(df[col].median())` |
| **Mean** | Data follows a **Normal Distribution** (bell curve) and has **no significant outliers**. | It uses all available data points and is the statistically correct center for symmetric data. | `df[col].fillna(df[col].mean())` |
| **Random** (Mean $\pm$ Std Dev) | You need to preserve the **variance** (spread) of the data, especially if many values are missing. | It avoids adding a single identical value, which artificially shrinks the variance. | `np.random.uniform(mean - std, mean + std, size)` |

To help you visualize the difference between these distributions:
<img width="3792" height="2744" alt="image" src="https://github.com/user-attachments/assets/1d6bf7f9-4fc6-420b-beab-07d84717665b" />

---

### 2. Categorical Data (Text or Ordinal)

For non-numeric data, you cannot calculate an average, so the methods focus on frequency or preservation of the 'missing' state.

| Statistical Measure | When to Use It | Why Choose It | How to Code (Pandas) |
| :--- | :--- | :--- | :--- |
| **Mode** | Missing percentage is **low** (e.g., less than 5%) and you want the simplest solution. | It represents the **most likely** existing value. | `df[col].fillna(df[col].mode()[0])` |
| **Arbitrary Value** | Missing percentage is **high** (e.g., over 10%) or the fact that the data is missing is itself important. | Filling with a unique string like **"Unknown"** or **"Missing"** preserves the information that the original data was unavailable, turning "missingness" into its own category. | `df[col].fillna('Missing')` |

---

### 🔑 Decision Checklist

1.  **Check Data Type:** Is the column numeric or categorical?
2.  **Check Missing %:** If the column is missing very little data (e.g., $< 2-3\%$), consider **dropping the rows** (`df.dropna(subset=[col])`) for that column instead of imputing, as the impact on analysis is negligible.
3.  **Plot (If Numeric):** Generate a **histogram** of the existing data. If it looks symmetrical (like a bell curve), use the **Mean**. If it is heavily skewed or shows clear outliers, use the **Median**.


