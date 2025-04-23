# Fraud_Transaction_Prediction

## Overview
This repository contains a machine learning project focused on detecting fraudulent financial transactions using advanced resampling techniques and classification algorithms.

- Tackles the challenge of highly imbalanced data (only 0.172% fraudulent transactions)
- Implements multiple resampling approaches (Undersampling, Oversampling, SMOTE, ADASYN)
- Compares performance of Decision Tree, XGBoost and Logistic Regression models

## Key Findings
- **Decision Tree with Oversampling** achieves 0.99 F1-Score
- XGBoost models consistently deliver strong performance across resampling techniques
- Demonstrates why accuracy is a poor metric for imbalanced fraud detection tasks

## Repository Contents
- Jupyter notebook with complete analysis and code
- Detailed visualizations of transaction patterns
- Feature engineering techniques for fraud detection
- Performance evaluation across multiple metrics

## Getting Started
Check the notebook to see the complete analysis process from data exploration to model evaluation.

---

### Technical Implementation Example Code
```python
# Import imbalace technique algorithims
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
```
```python
# Oversampled
random_state = 42

ros = RandomOverSampler(random_state=random_state)
X_res, y_res = ros.fit_resample(X_train, y_train)

print('Resampled dataset shape %s' % Counter(y_res))

X_train_over = X_res
y_train_over = y_res
X_test_over, y_test_over = X_test, y_test

print("X_train_over - ",X_train_over.shape)
print("y_train_over - ",y_train_over.shape)
print("X_test_over - ",X_test_over.shape)
print("y_test_over - ",y_test_over.shape)
```
