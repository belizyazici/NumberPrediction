# Predictive Modeling with Ensemble Learning on Process Data

## Overview

This project aims to build a robust predictive model on a dataset containing process-based features (x1–x6) to predict a target variable (Y). The main goal is to increase accuracy and generalization by combining multiple preprocessing techniques, feature engineering steps, and ensemble learning methods.

---

## Dataset Description

The dataset contains six numeric features (x1–x6) and a target variable Y. The samples are split as follows:

* **Training samples**: SampleNo ≤ 100
* **Testing samples**: SampleNo > 100

---

## Methodology

### 1. Initial Setup

* The project initially began with **Random Forest** modeling to evaluate baseline performance.

### 2. Outlier Detection

* Applied **IQR (Interquartile Range)** method to detect and remove outliers from training data.

### 3. Feature Engineering

Several new features were added to enrich the dataset:

* `x1_x2_ratio` = x1 / (x2 + 1e-5)
* `x1_x3_interact` = x1 \* x3
* `log_x1` = log(x1 + 1e-5)
* `x4_squared` = x4²

### 4. Data Augmentation

* Introduced slight Gaussian noise to the dataset for **data augmentation**, helping reduce overfitting and improve model generalization.

### 5. Normalization

* Applied **QuantileTransformer** to normalize both features and target (Y) distribution.

### 6. Feature Selection

* Used **LGBMRegressor** with `SelectFromModel` to reduce dimensionality and keep the most important features.

### 7. Hyperparameter Optimization

* Performed **RandomizedSearchCV** on XGBoost for tuning hyperparameters.

### 8. Ensemble Learning (Final Model)

* Combined four models into a **Voting Regressor**:

  * XGBoost (best model from tuning)
  * Random Forest
  * CatBoost
  * LightGBM

### 9. Cross-Validation and Evaluation

* Used 10-Fold Cross-Validation to assess generalization.
* Measured training RMSE, R², and overfitting gap (Training vs CV error).

### 10. Final Predictions

* Made predictions on the test set.
* Applied inverse transformation to get final Y values.
* Saved predictions to `final_predictions.xlsx`.

---

## Results

* **Training RMSE** and **R²** show excellent performance.
* Cross-validation results indicate the model generalizes well.
* Predictions were successfully saved in Excel format.

---

## Files

* `NumberPrediction.py` → Main implementation file.
* `ProjectDataset.xlsx` → Raw dataset.
* `final_predictions.xlsx` → Final prediction results.

---

## Requirements

* Python 3.8+
* pandas, numpy
* scikit-learn
* xgboost
* lightgbm
* catboost



