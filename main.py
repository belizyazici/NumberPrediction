import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso


file = pd.ExcelFile("ProjectDataset.xlsx")
# print(file.sheet_names)


data_df = file.parse('Data')

train_df = data_df.iloc[:100]  # first 100 for train
pred_df = data_df.iloc[100:]  # last 20 for prediction

feature_col = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

X_train = train_df[feature_col]
y_train = train_df['Y']

X_test = pred_df[feature_col]
"""
print("*************** First 5 data ****************")
print(data_df.head())
print("\n************** General Info *****************")
print(data_df.info())
print("\n************* Numerical Summarizing *************")
print(data_df.describe())

print("\n************* Observations which have missing data *************")
print(data_df.tail(20))


train_df = data_df.iloc[:100]  # first 100 for train
pred_df = data_df.iloc[100:]  # last 20 for prediction

feature_col = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

X_train = train_df[feature_col]
y_train = train_df['Y']

X_test = pred_df[feature_col]

print(X_train.shape)  # (100, 6) 100 örnek 6 feature
print(y_train.shape)  # (100,) 100 Y value

print(X_test.shape)  # (20, 6)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)


# Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)

plt.figure(figsize=(12, 6))
plt.plot(pred_df['SampleNo'], lr_preds, label='LinearRegression', marker='o')
plt.plot(pred_df['SampleNo'], rf_preds, label='RandomForest', marker='x')
plt.plot(pred_df['SampleNo'], dt_preds, label='DecisionTree', marker='s')
plt.title('Y Predictions of Each Model')
plt.xlabel('Sample No')
plt.ylabel('Predicted Y')
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.show()


# Her model için train verisinde tahmin
#lr_train_pred = lr.predict(X_train)
rf_train_pred = rf.predict(X_train)
#dt_train_pred = dt.predict(X_train)


# Linear Regression
lr_mse = mean_squared_error(y_train, lr_train_pred)
lr_r2 = r2_score(y_train, lr_train_pred)

# Random Forest
rf_mse = mean_squared_error(y_train, rf_train_pred)
rf_r2 = r2_score(y_train, rf_train_pred)


# Decision Tree
dt_mse = mean_squared_error(y_train, dt_train_pred)
dt_r2 = r2_score(y_train, dt_train_pred)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Train verisindeki performans
lasso_train_pred = lasso.predict(X_train)
lasso_mse = mean_squared_error(y_train, lasso_train_pred)
lasso_r2 = r2_score(y_train, lasso_train_pred)

# Test (prediction) verisinde tahmin
lasso_preds = lasso.predict(X_test)


print("Model Performances (Train Data):")
#print(f"Linear Regression  → MSE: {lr_mse:.2f},    R²: {lr_r2:.4f}")  # MSE: 3130100.60, R²: 0.2938 Kötü — veriyi iyi açıklayamıyor
print(f"Random Forest      → MSE: {rf_mse:.2f},    R²: {rf_r2:.4f}")  # MSE: 476041.37, R²: 0.8926 	Güçlü — karmaşık ilişkileri yakalıyor
#print(f"Decision Tree      → MSE: {dt_mse:.2f},    R²: {dt_r2:.4f}")  # MSE: 0.00, R²: 1.0000 Mükemmel ama ezberliyor (overfit)
#print(f"Lasso Regression   → MSE: {lasso_mse:.2f}, R²: {lasso_r2:.4f}")  # MSE: 3130100.60, R²: 0.2938 	Aynısı — L1 cezasına rağmen fark yaratmamış

print("Gerçek Y değerlerinin ortalaması:", y_train.mean())
print("Gerçek Y değerlerinin std'si:", y_train.std())


from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_train, rf.predict(X_train)))
print(f"Random Forest RMSE: {rmse:.2f}")

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Hiperparametreler
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Model
rf = RandomForestRegressor(random_state=42)

# GridSearch ile en iyisini bul
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# En iyi model
best_rf = grid_search.best_estimator_

print("En iyi parametreler:", grid_search.best_params_)

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Tahmin ve metrikler
y_pred = best_rf.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, y_pred)

print(f"Tuned Random Forest → MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
"""
print("******************* XG BOOST *****************************")


from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

# Modeli oluştur
xgb_model = XGBRegressor(random_state=42)

# Train setinde modeli eğit
xgb_model.fit(X_train, y_train)

# Train verisinde tahmin yap
y_train_pred_xgb = xgb_model.predict(X_train)

# Performansı ölçelim
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

mse_xgb = mean_squared_error(y_train, y_train_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_train, y_train_pred_xgb)

print(f"XGBoost → MSE: {mse_xgb:.2f}, RMSE: {rmse_xgb:.2f}, R²: {r2_xgb:.4f}")


xgb_tuned = XGBRegressor(
    max_depth=3,
    n_estimators=100,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)

# Modeli eğit
xgb_tuned.fit(X_train, y_train)

# Tahmin
y_train_pred_xgb_tuned = xgb_tuned.predict(X_train)

# Performans ölçümü
mse_xgb_tuned = mean_squared_error(y_train, y_train_pred_xgb_tuned)
rmse_xgb_tuned = np.sqrt(mse_xgb_tuned)
r2_xgb_tuned = r2_score(y_train, y_train_pred_xgb_tuned)

print(f"Tuned XGBoost → MSE: {mse_xgb_tuned:.2f}, RMSE: {rmse_xgb_tuned:.2f}, R²: {r2_xgb_tuned:.4f}")
