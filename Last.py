import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from catboost import CatBoostRegressor
from sklearn.ensemble import IsolationForest
import shap


file = pd.ExcelFile("ProjectDataset.xlsx")
data_df = file.parse('Data')


train_df = data_df.iloc[:100]
pred_df = data_df.iloc[100:]


feature_col = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
X = train_df[feature_col]
y = train_df['Y']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


catboost_model = CatBoostRegressor(verbose=0, random_state=42, depth=4, l2_leaf_reg=10)
catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20, verbose=0)
y_pred = catboost_model.predict(X)
print(f"CatBoost MSE: {mean_squared_error(y, y_pred):.2f}, "
      f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.2f}, "
      f"R²: {r2_score(y, y_pred):.4f}")


def isolation_forest_outliers(df, cols, contamination=0.05):
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(df[cols])
    return df[outliers == 1]


def add_non_linear_features(df):
    df = df.copy()
    for col in df.columns:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
        df[f'{col}_log'] = np.log1p(np.abs(df[col]))
    return df


def get_trained_model_and_shap(X, y):
    model = CatBoostRegressor(verbose=0, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20, verbose=0)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    return model, shap_values


def get_top_features_from_shap(X, shap_values, top_n=20):
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    shap_df = pd.DataFrame({'feature': X.columns, 'importance': shap_importance})
    top_features = shap_df.sort_values('importance', ascending=False).head(top_n)['feature'].tolist()
    return top_features


def train_and_evaluate(model, model_name, X_final, y_final):
    X_train, X_val, y_train, y_val = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20, verbose=0)
    y_pred = model.predict(X_final)
    mse = mean_squared_error(y_final, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_final, y_pred)
    print(f"\n{model_name} Model MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_rmse = np.sqrt(-cross_val_score(model, X_final, y_final, scoring='neg_mean_squared_error', cv=kf))
    print(f"{model_name} 10-Fold CV RMSE: {cv_rmse}")
    print(f"Mean RMSE: {cv_rmse.mean():.2f}, Std Dev: {cv_rmse.std():.2f}")


train_df_iforest = isolation_forest_outliers(train_df, feature_col)
X_iforest = train_df_iforest[feature_col]
y_iforest = train_df_iforest['Y']
X_iforest_nl = add_non_linear_features(X_iforest)
catboost_final_model, shap_values_final = get_trained_model_and_shap(X_iforest_nl, y_iforest)
top_features_iforest = get_top_features_from_shap(X_iforest_nl, shap_values_final)
X_iforest_final = X_iforest_nl[top_features_iforest]

train_and_evaluate(catboost_final_model, "CatBoost with Isolation Forest", X_iforest_final, y_iforest)


g = shap.plots.beeswarm(shap_values_final, show=False)


def augment_data_scaled(X, y, noise_level=0.01, multiplier=2, seed=42):
    np.random.seed(seed)  # Sabit tohum
    augmented_X = [X]
    augmented_y = [y]
    stds = X.std()
    for _ in range(multiplier):
        noise = np.random.normal(0, noise_level, X.shape) * stds.values
        X_aug = X + noise
        augmented_X.append(X_aug)
        augmented_y.append(y)
    X_total = pd.concat(augmented_X, ignore_index=True)
    y_total = pd.concat([pd.Series(a) for a in augmented_y], ignore_index=True)
    return X_total, y_total


X_aug_iforest, y_aug_iforest = augment_data_scaled(X_iforest_final, y_iforest, noise_level=0.02, multiplier=2, seed=42)

train_and_evaluate(CatBoostRegressor(verbose=0, random_state=42),
                   "CatBoost with Isolation Forest + Augmentation",
                   X_aug_iforest, y_aug_iforest)


from sklearn.model_selection import RandomizedSearchCV, KFold

"""
param_dist = {
    'iterations': [500, 800, 1000, 1500],
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'bagging_temperature': [0.1, 0.5, 1.0],
    'random_strength': [0.5, 1, 1.5, 2],
    'boosting_type': ['Ordered', 'Plain']
}


cv = KFold(n_splits=5, shuffle=True, random_state=42)


random_search = RandomizedSearchCV(
    estimator=CatBoostRegressor(verbose=0, random_state=42),
    param_distributions=param_dist,
    n_iter=30,
    scoring='neg_root_mean_squared_error',
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=42
)


random_search.fit(X_aug_iforest, y_aug_iforest)


best_model = random_search.best_estimator_
train_and_evaluate(best_model, "Tuned CatBoost with Isolation Forest + Augmentation", X_aug_iforest, y_aug_iforest)


print("\nBest Hyperparameters:", random_search.best_params_)

"""

best_model = CatBoostRegressor(
    random_strength=0.5,
    learning_rate=0.03,
    l2_leaf_reg=1,
    iterations=1500,
    depth=8,
    boosting_type='Ordered',
    bagging_temperature=0.5,
    verbose=0,
    random_state=42
)

train_and_evaluate(
    best_model,
    "Tuned CatBoost with Isolation Forest + Augmentation",
    X_aug_iforest,
    y_aug_iforest
)
