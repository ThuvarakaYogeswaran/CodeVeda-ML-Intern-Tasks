# Codveda Technologies — Machine Learning Internship Portfolio

### A complete collection of all Machine Learning tasks completed during the Codveda Technologies internship (Levels 1–3)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Tasks](https://img.shields.io/badge/Tasks-6%2F6-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A hands-on Machine Learning internship portfolio covering the full
> supervised-learning pipeline — from data preprocessing to advanced ensemble
> methods and kernel-based classifiers. Every task includes a reproducible
> Jupyter notebook, a detailed README, and visualizations.

**🎯 Domain:** Machine Learning · **📚 Levels completed:** 3/3 · **✅ Tasks completed:** 6/6

---

## 📌 Table of Contents

- [Overview](#overview)
- [Skills Demonstrated](#skills-demonstrated)
- [Task Summary Table](#task-summary-table)
- [Level 1 — Basic](#level-1--basic)
- [Level 2 — Intermediate](#level-2--intermediate)
- [Level 3 — Advanced](#level-3--advanced)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [How to Use This Repo](#how-to-use-this-repo)
- [Key Learnings](#key-learnings)
- [Internship Information](#internship-information)

---

## Overview

This repository consolidates **6 machine learning tasks** across **3 internship levels** covering the fundamental algorithms and workflows used in modern ML:

- **Data preprocessing** — cleaning, encoding, scaling, splitting
- **Classification** — KNN, logistic regression, decision trees, random forests, SVMs
- **Model evaluation** — accuracy, precision, recall, F1, ROC/AUC, confusion matrices, cross-validation
- **Model interpretation** — odds ratios, feature importances, decision boundaries, tree structure

Each task lives in its own folder with its own dedicated README, notebook, dataset, and output images.

---

## Skills Demonstrated

| Category | Skills |
|---|---|
| **Data** | Loading, cleaning, missing-value handling, encoding, standardization, stratified splitting |
| **Classification** | KNN, logistic regression, decision trees, random forests, SVM |
| **Model Selection** | GridSearchCV, StratifiedKFold cross-validation |
| **Evaluation** | Accuracy, precision, recall, F1, confusion matrix, ROC/AUC |
| **Interpretation** | Odds ratios, feature importance, decision tree structure, decision boundaries |
| **Visualization** | Matplotlib, Seaborn |
| **Engineering** | Jupyter notebooks, reproducible pipelines, GitHub portfolio |

---

## Task Summary Table

| # | Level | Task | Algorithm | Dataset | Key Metric | 
|---|---|---|---|---------|------------|
| 1 | 1 — Basic | Data Preprocessing | — | Iris | Clean dataset, 80/20 split | 
| 2 | 1 — Basic | KNN Classifier | K-Nearest Neighbors | Iris | Accuracy 96.67% | 
| 3 | 2 — Intermediate | Logistic Regression | Logistic Regression | Iris (binary) | Accuracy 100%, AUC 1.0 | 
| 4 | 2 — Intermediate | Decision Tree | Decision Tree (pruned) | Boston Housing (3-class) | Accuracy ~82% | 
| 5 | 3 — Advanced | Random Forest | Random Forest (tuned) | Boston Housing (3-class) | Accuracy ~85% | 
| 6 | 3 — Advanced | SVM Classifier | Support Vector Machine | Iris (versicolor vs virginica) | Accuracy ~95%, AUC ~0.97 | 

> **Note:** The Codveda internship requires **2 tasks per level**. All 6 tasks shown above satisfy that requirement.

---

## Level 1 — Basic

### 📁 Task 1 — Data Preprocessing for Machine Learning

**Goal:** Prepare a raw dataset for ML by handling missing values, encoding categorical features, and scaling numerical features.

- **Dataset:** Iris (150 samples, 4 features, 3 classes)
- **Steps:** Null check → LabelEncoder → StandardScaler → 80/20 stratified split
- **Output:** Clean, model-ready dataset with balanced class distribution across train and test
- **Key insight:** Preprocessing often has more impact on model performance than algorithm choice
- **Report:** [`Level-1-Basic/Task-1-Data-Preprocessing/README.md`](Level-1-Basic/Task-1-Data-Preprocessing/README.md)

### 📁 Task 2 — K-Nearest Neighbors (KNN) Classifier

**Goal:** Classify data points using KNN across multiple K values.

- **Dataset:** Iris
- **K values tested:** 1, 3, 5, 7, 9, 11, 15, 21
- **Best accuracy:** 96.67% (29/30 test samples correct)
- **Key insight:** Feature scaling is essential for distance-based algorithms; accuracy plateaus at K ≥ 7, showing the model is robust to K choice
- **Report:** [`Level-1-Basic/Task-3-KNN-Classifier/README.md`](Level-1-Basic/Task-3-KNN-Classifier/README.md)

---

## Level 2 — Intermediate

### 📁 Task 1 — Logistic Regression for Binary Classification

**Goal:** Predict a binary outcome using logistic regression.

- **Dataset:** Iris (setosa vs rest)
- **Metrics:** Accuracy 100%, AUC 1.0
- **Key insight:** Perfect score is expected — setosa is linearly separable. Odds-ratio analysis shows `petal_length` and `petal_width` are strongest negative predictors of setosa membership, while `sepal_width` is a positive predictor
- **Report:** [`Level-2-Intermediate/Task-1-Logistic-Regression/README.md`](Level-2-Intermediate/Task-1-Logistic-Regression/README.md)

### 📁 Task 2 — Decision Tree for Classification

**Goal:** Classify categorical outcomes using a decision tree, with pruning.

- **Dataset:** Boston Housing (binned into Low / Medium / High)
- **Unpruned:** 100% train / 72% test — classic overfitting
- **Pruned (max_depth=4):** ~82% test accuracy
- **Key insight:** Pruning closed the train/test gap by ~10 percentage points. Top features: `LSTAT` (socioeconomic status) and `RM` (rooms per dwelling)
- **Report:** [`Level-2-Intermediate/Task-2-Decision-Tree/README.md`](Level-2-Intermediate/Task-2-Decision-Tree/README.md)

---

## Level 3 — Advanced

### 📁 Task 1 — Random Forest Classifier

**Goal:** Build a tuned ensemble classifier and compare against a single tree.

- **Dataset:** Boston Housing (binned 3-class)
- **Tuning:** GridSearchCV over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`
- **Best accuracy:** ~85% (3 points above the pruned Decision Tree)
- **Key insight:** Bagging reduces variance on tabular data. Feature importance again ranks `LSTAT` and `RM` at the top — matching domain knowledge
- **Report:** [`Level-3-Advanced/Task-1-Random-Forest/README.md`](Level-3-Advanced/Task-1-Random-Forest/README.md)

### 📁 Task 2 — Support Vector Machine (SVM) for Classification

**Goal:** Binary classification with kernel comparison and decision-boundary visualization.

- **Dataset:** Iris (versicolor vs virginica — the harder pair)
- **Kernels compared:** linear, RBF, polynomial
- **Best accuracy:** ~95% (RBF kernel, C=10, gamma=0.1), AUC ~0.97
- **Key insight:** RBF outperforms linear by ~5 points on this non-separable pair. The 2D decision-boundary plot makes the difference visually obvious
- **Report:** [`Level-3-Advanced/Task-2-SVM/README.md`](Level-3-Advanced/Task-2-SVM/README.md)

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.10+** | Core language |
| **pandas** | Data loading, manipulation, exploration |
| **NumPy** | Numeric operations |
| **scikit-learn** | Models, preprocessing, metrics, cross-validation |
| **Matplotlib** | Custom plots, decision boundaries |
| **Seaborn** | Statistical visualizations, confusion matrix heatmaps |
| **Jupyter Notebook** | Development and storytelling environment |

---

## Repository Structure

```text
Codveda-ML-Internship/
│
├── README.md                                  ← you are here
│
├── Level-1-Basic/
│   ├── Task-1-Data-Preprocessing/
│   │   ├── Data_Preprocessing.ipynb
│   │   ├── iris.csv
│   │   ├── README.md
│   │   └── images/
│   └── Task-3-KNN-Classifier/
│       ├── KNN_Iris_Classification.ipynb
│       ├── iris.csv
│       ├── README.md
│       └── images/
│
├── Level-2-Intermediate/
│   ├── Task-1-Logistic-Regression/
│   │   ├── Logistic_Regression_Iris.ipynb
│   │   ├── iris.csv
│   │   ├── README.md
│   │   └── images/
│   └── Task-2-Decision-Tree/
│       ├── Decision_Tree_Boston.ipynb
│       ├── house_prediction.csv
│       ├── README.md
│       └── images/
│
└── Level-3-Advanced/
    ├── Task-1-Random-Forest/
    │   ├── Random_Forest_Boston.ipynb
    │   ├── house_prediction.csv
    │   ├── README.md
    │   └── images/
    └── Task-2-SVM/
        ├── SVM_Iris_Classification.ipynb
        ├── iris.csv
        ├── README.md
        └── images/
```

> **Note:** Task folders keep their original numbering from the internship brief (Task 1, Task 3 in Level 1; Task 1, Task 2 in Levels 2 and 3). Only the two required tasks per level are included.

---


## Key Learnings

Across all 6 tasks, the following themes emerged:

1. **Preprocessing matters more than the algorithm**  
   Scaling alone boosted KNN accuracy to 96.67% on Iris — without scaling, distance-based models lose 10+ percentage points.

2. **Model complexity is a tradeoff, not a goal**  
   An unpruned Decision Tree scored 100% on training data but only 72% on the test set. Pruning lifted test accuracy by 10 percentage points.

3. **Ensembles beat single models — usually**  
   Random Forest (~85%) outperformed a single pruned Decision Tree (~82%) on Boston Housing by averaging away individual tree variance.

4. **Kernel choice is everything for SVMs**  
   Linear vs RBF made a ~5-point difference on versicolor vs virginica — a class pair that is not linearly separable.

5. **Perfect scores deserve scrutiny**  
   Logistic regression hitting 100% on setosa-vs-rest was legitimate (linearly separable classes), not a bug. Explaining *why* a score is perfect is more valuable than chasing it.

6. **Interpretability ≠ accuracy**  
   Logistic regression's odds ratios and Decision Tree's structure give clear "why" answers, while Random Forest trades some interpretability for accuracy. Choosing the right model depends on the use case.

7. **Cross-validation exposes fragility**  
   A single train/test split can be lucky. 5-fold CV revealed that Random Forest had ~0.014 std across folds — genuinely stable — while SVM showed one weaker fold, pointing to the harder versicolor/virginica boundary.

---

## Internship Information

**Organization:** Codveda Technologies  
**Domain:** Machine Learning  
**Levels Completed:** Level 1 (Basic), Level 2 (Intermediate), Level 3 (Advanced)  
**Duration:** 1 month  
**Requirement:** 2 tasks per level — **all 6 required tasks completed**

---

