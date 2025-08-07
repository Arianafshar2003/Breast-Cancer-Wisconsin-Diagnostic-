# 🎗️ Breast Cancer Wisconsin (Diagnostic) Classification
### Machine Learning Project – Classification Problem  
---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Objectives](#objectives)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Selection](#feature-selection)
6. [Modeling](#modeling)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Presentation](#presentation)
10. [How to Run](#how-to-run)
11. [Acknowledgements](#acknowledgements)
12. [License](#license)
13. [توضیحات فارسی](#توضیحات-فارسی)

---

## Project Overview

This project applies **machine learning classification** methods to the **Breast Cancer Wisconsin (Diagnostic)** dataset to distinguish between **malignant** and **benign** tumors based on extracted cell nucleus features from digitized images.  

The pipeline includes:
- EDA (Exploratory Data Analysis)
- Preprocessing
- Feature selection using 5 different methods
- Model training & hyperparameter tuning
- Performance evaluation using multiple metrics

---

## Dataset

[- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  
- **Instances:** 569 samples  
- **Features:** 30 numerical features describing cell nuclei (radius, texture, perimeter, smoothness, etc.)  
- **Target variable:** `target` → 0 = malignant, 1 = benign  
- **No missing values** in the original dataset  

---

## Objectives

As outlined in the presentation, the goals were to:
1. **Train and compare** different classification models using various evaluation metrics.  
2. **Perform feature selection** with 5 different methods to analyze their effect on model performance.  
3. **Evaluate** using cross-validation for reliable performance estimates.  
4. **Assess** the impact of feature selection on results.  
5. **Analyze performance** using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

---

## Data Preprocessing

- Checked for missing/null values (none found)
- Label distribution analysis and visualization
- Split into training and testing sets
- **StandardScaler** for feature normalization
- Optional: Outlier removal based on feature bounds

---

## Feature Selection

Five methods applied:
1. Mutual Information
2. Chi-Square
3. ANOVA F-value
4. Random Forest Feature Importance
5. Recursive Feature Elimination (RFE)

---

## Modeling

Models compared:
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Classifier (SVC)  
- Decision Tree  
- Random Forest  
- Gaussian Naive Bayes  
- Gradient Boosting  
- Bagging  
- Voting Classifier (ensemble)  

Hyperparameter tuning via **GridSearchCV** (3-fold cross-validation).

---

## Evaluation

Metrics:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC Score  
- Confusion Matrix  
- ROC curves visualized for top models

---

## Results

- Achieved high accuracy (>95%) for top-performing models (Random Forest, Gradient Boosting, VotingClassifier)  
- ROC-AUC close to 1.0 for the best models, indicating excellent separation capability  
- Feature selection slightly improved performance in some models and reduced training time  
- Ensemble methods provided the most stable performance across different feature sets

---

## Presentation

The presentation detailing all steps, analysis, and comparisons is included here:  
[`presentation.pdf`](presentation.pdf)

---

## How to Run

1. **Clone this repository:**
    ```bash
    git clone https://github.com/Arianafshar2003/Breast-Cancer-Wisconsin-Classification.git
    cd Breast-Cancer-Wisconsin-Classification
    ```
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Launch the notebook:**
    ```bash
    jupyter notebook code.ipynb
    ```
4. Run all cells to execute the classification workflow.

---


## توضیحات فارسی

در این پروژه، دیتاست Breast Cancer Wisconsin (Diagnostic) برای شناسایی تومورهای بدخیم و خوش‌خیم به کمک الگوریتم‌های یادگیری ماشین استفاده شده است. مراحل شامل:
1. تحلیل اکتشافی داده‌ها (EDA)  
2. پیش‌پردازش و نرمال‌سازی ویژگی‌ها  
3. انتخاب ویژگی با ۵ روش مختلف  
4. آموزش و مقایسه مدل‌های طبقه‌بندی  
5. ارزیابی با شاخص‌هایی مانند Accuracy، Precision، Recall، F1 و ROC-AUC  
خلاصه و تحلیل نتایج در فایل [presentation.pdf](presentation.pdf) موجود است.

---
