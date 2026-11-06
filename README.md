# Datheon Machine Learning & Data Science Internship

Portfolio of tasks completed for the Datheon ML & Data Science Internship.

## Task 1: Multi-Class Benchmark Classification Pipeline (Iris)

End-to-end supervised classification pipeline built with Scikit-Learn on the Iris dataset.

### What it covers
- Exploratory data analysis with Pandas and Seaborn (pairplot, class balance)
- Feature scaling with StandardScaler
- Stratified 80/20 train/test split
- Models: Logistic Regression and Decision Tree
- Metrics: accuracy, precision, recall and F1-score (macro-averaged)
- Confusion matrices and decision boundary visualization

### Results
| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.933 | 0.933 | 0.933 | 0.933 |
| Decision Tree | 0.967 | 0.970 | 0.967 | 0.967 |

## Task 2: Real Estate Housing Price Prediction (Ridge & Lasso Regression)

Continuous regression modeling on the California Housing dataset (20,640 samples).

### What it covers
- Exploratory data analysis (target distribution, correlation matrix)
- Feature engineering: BedroomRatio, RoomsPerPerson, LogPopulation
- StandardScaler and 80/20 train/test split
- Models: Linear Regression (baseline), Ridge and Lasso with alpha tuned by 5-fold cross-validation
- Metrics: MAE, RMSE and R2
- Residual analysis and coefficient comparison

### Results
| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Linear Regression | 0.486 | 0.675 | 0.652 |
| Ridge (alpha = 62.10) | 0.486 | 0.676 | 0.651 |
| Lasso (alpha = 0.0028) | 0.487 | 0.676 | 0.651 |

Lasso removed one feature (Population), which is redundant with the engineered LogPopulation feature.

## Files
- `task1_classification_iris.ipynb`: Task 1 notebook
- `task2_housing_regression.ipynb`: Task 2 notebook

## How to run
pip install pandas scikit-learn seaborn matplotlib jupyter