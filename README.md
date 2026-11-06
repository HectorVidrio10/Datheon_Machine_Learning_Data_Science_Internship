# Datheon Machine Learning & Data Science Internship

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

### Files
- `task1_classification_iris.ipynb`: notebook with the full analysis

### How to run
pip install pandas scikit-learn seaborn matplotlib jupyter