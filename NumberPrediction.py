import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings


warnings.filterwarnings('ignore')


df = pd.read_excel("ProjectDataset.xlsx")
train_df = df[df["SampleNo"] <= 100].copy()
test_df = df[df["SampleNo"] > 100].copy()


X = train_df.drop(columns=["Y", "SampleNo"])
y = train_df["Y"]
X_test = test_df.drop(columns=["Y", "SampleNo"])


Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
X, y = X[mask], y[mask]


X["x1_x2_ratio"] = X["x1"] / (X["x2"] + 1e-5)
X["x1_x3_interact"] = X["x1"] * X["x3"]
X["log_x1"] = np.log(X["x1"] + 1e-5)
X["x4_squared"] = X["x4"] ** 2

X_test["x1_x2_ratio"] = X_test["x1"] / (X_test["x2"] + 1e-5)
X_test["x1_x3_interact"] = X_test["x1"] * X_test["x3"]
X_test["log_x1"] = np.log(X_test["x1"] + 1e-5)
X_test["x4_squared"] = X_test["x4"] ** 2


def data_augmentation(X, y, noise_level=0.01):
    X_aug = X + np.random.normal(0, noise_level, X.shape)
    return pd.concat([X, X_aug]), pd.concat([y, y])


X, y = data_augmentation(X, y)


scaler = QuantileTransformer(output_distribution='normal')
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

qt = QuantileTransformer(output_distribution='normal')
y_trans = qt.fit_transform(y.values.reshape(-1, 1)).ravel()


lgb_selector = lgb.LGBMRegressor()
lgb_selector.fit(X_scaled, y_trans)
model_selector = SelectFromModel(lgb_selector, threshold="median", prefit=True)
X_selected = model_selector.transform(X_scaled)
X_test_selected = model_selector.transform(X_test_scaled)


xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
param_dist = {
    "n_estimators": [100, 150, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 9],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, scoring='neg_root_mean_squared_error',
                                   n_iter=15, cv=5, random_state=42)
random_search.fit(X_selected, y_trans)
best_xgb = random_search.best_estimator_


base_models = [
    ('xgb', best_xgb),
    ('rf', RandomForestRegressor(n_estimators=150, random_state=42)),
    ('cat', cb.CatBoostRegressor(verbose=0)),
    ('lgb', lgb.LGBMRegressor())
]

voting_model = VotingRegressor(estimators=base_models)
voting_model.fit(X_selected, y_trans)


cv = KFold(n_splits=10, shuffle=True, random_state=42)

cv_scores = cross_val_score(voting_model, X_selected, y_trans, scoring='neg_root_mean_squared_error', cv=cv)

train_preds = voting_model.predict(X_selected)
train_rmse = np.sqrt(mean_squared_error(y_trans, train_preds))
r2 = r2_score(y_trans, train_preds)
overfit_disrepancy = train_rmse - (-cv_scores.mean())

print(f"RMSE on the training set: {train_rmse:.4f}")
print(f"R2: {r2:.4f}")
print(f"Average RMSE from Cross-Validation: {-cv_scores.mean():.4f}")
print(f"Discrepancy Between Training and CV (Overfitting Indicator): {overfit_disrepancy:.4f}")

test_preds_trans = voting_model.predict(X_test_selected)
test_preds_original = qt.inverse_transform(test_preds_trans.reshape(-1, 1)).ravel()
test_df["Predicted_Y"] = test_preds_original
test_df[["Predicted_Y"]].to_excel("final_predictions.xlsx", index=False, header=False)

print("Predictions saved in final_predictions.xlsx file")
