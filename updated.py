import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from catboost import CatBoostRegressor
from sklearn.ensemble import IsolationForest
import shap


# Veriyi oku
file = pd.ExcelFile("ProjectDataset.xlsx")
data_df = file.parse('Data')

# Eğitim ve tahmin setlerini ayır
train_df = data_df.iloc[:100]
pred_df = data_df.iloc[100:]

# Özellik sütunları
feature_col = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
X_train = train_df[feature_col]
y_train = train_df['Y']

# 1. CatBoost Modeli
catboost_model = CatBoostRegressor(verbose=0, random_state=42)
catboost_model.fit(X_train, y_train)
y_pred = catboost_model.predict(X_train)
print(f"CatBoost → MSE: {mean_squared_error(y_train, y_pred):.2f}, "
      f"RMSE: {np.sqrt(mean_squared_error(y_train, y_pred)):.2f}, "
      f"R²: {r2_score(y_train, y_pred):.4f}")

# 2. Aykırı Değer Temizliği (Isolation Forest)
def isolation_forest_outliers(df, cols, contamination=0.05):
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(df[cols])
    return df[outliers == 1]  # Yalnızca normal olan verileri döndürür

train_df_iforest = isolation_forest_outliers(train_df, feature_col)

print(f"Orijinal veri boyutu: {train_df.shape[0]}")
print(f"Isolation Forest ile temizlenmiş veri boyutu: {train_df_iforest.shape[0]}")

# 3. Non-lineer öznitelik üretme
def add_non_linear_features(df):
    df = df.copy()
    for col in df.columns:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
        df[f'{col}_log'] = np.log1p(np.abs(df[col]))
    return df

# 4. SHAP ile en önemli öznitelikleri seçme
def select_top_features_with_shap(X, y, top_n=20):
    model = CatBoostRegressor(verbose=0, random_state=42).fit(X, y)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    shap_df = pd.DataFrame({'feature': X.columns, 'importance': shap_importance})
    top_features = shap_df.sort_values('importance', ascending=False).head(top_n)['feature'].tolist()
    return top_features

# CatBoost Modelini Eğit ve Değerlendir
def train_and_evaluate(model, model_name, X_final, y_final):
    model.fit(X_final, y_final)
    y_pred = model.predict(X_final)

    mse = mean_squared_error(y_final, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_final, y_pred)

    print(f"\n{model_name} Model → MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

    # 10-Fold CV
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_rmse = np.sqrt(-cross_val_score(model, X_final, y_final, scoring='neg_mean_squared_error', cv=kf))
    print(f"{model_name} 10-Fold CV RMSE: {cv_rmse}")
    print(f"Mean RMSE: {cv_rmse.mean():.2f}, Std Dev: {cv_rmse.std():.2f}")

# 5. Model Verisi (Isolation Forest temizlenmiş veri)
X_iforest = train_df_iforest[feature_col]
y_iforest = train_df_iforest['Y']
X_iforest_nl = add_non_linear_features(X_iforest)
top_features_iforest = select_top_features_with_shap(X_iforest_nl, y_iforest)
X_iforest_final = X_iforest_nl[top_features_iforest]

# CatBoost modelini eğit
train_and_evaluate(CatBoostRegressor(verbose=0, random_state=42), "CatBoost with Isolation Forest", X_iforest_final, y_iforest)

# Boxplot (Aykırı Değerler için)
sns.boxplot(data=train_df[feature_col])
plt.title("Boxplot of Features (Orijinal Veri)")
# plt.show()

# Tahmin vs Gerçek Y
catboost_final = CatBoostRegressor(verbose=0, random_state=42)
catboost_final.fit(X_iforest_final, y_iforest)
y_pred_final = catboost_final.predict(X_iforest_final)

plt.figure(figsize=(10, 6))
plt.plot(y_iforest.values, label='Gerçek Y', marker='o')
plt.plot(y_pred_final, label='Tahmin Y', marker='x')
plt.legend()
plt.title("Final Model Tahmin vs Gerçek (Isolation Forest Temizlenmiş Veri)")
plt.xlabel("Veri Nokası")
plt.ylabel("Y")
# plt.show()

# SHAP Analizi
explainer = shap.Explainer(catboost_final, X_iforest_final)
shap_values = explainer(X_iforest_final)
shap.summary_plot(shap_values, X_iforest_final, show=False)

# Öznitelik Önem Grafiği
importances = catboost_final.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_iforest_final.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title("Final Model - Feature Importance")
plt.tight_layout()
#plt.show()

# ===============================
# CatBoost + Isolation Forest + Augmentasyon
# ===============================
def augment_data(X, y, noise_level=0.01, multiplier=2):
    augmented_X = [X]
    augmented_y = [y]

    for _ in range(multiplier):
        noise = np.random.normal(0, noise_level, X.shape)
        X_aug = X + noise
        augmented_X.append(X_aug)
        augmented_y.append(y)

    X_total = pd.concat([pd.DataFrame(a, columns=X.columns) for a in augmented_X], ignore_index=True)
    y_total = pd.concat([pd.Series(a) for a in augmented_y], ignore_index=True)

    return X_total, y_total

# Augment edilmiş veri ile model eğitimi
X_aug_iforest, y_aug_iforest = augment_data(X_iforest_final, y_iforest, noise_level=0.02, multiplier=2)

train_and_evaluate(CatBoostRegressor(verbose=0, random_state=42),
                   "CatBoost with Isolation Forest + Augmentation",
                   X_aug_iforest, y_aug_iforest)

print(f"\nIsolation Forest sonrası orijinal veri sayısı: {y_iforest.shape[0]}")
print(f"Augmentation sonrası toplam veri sayısı: {y_aug_iforest.shape[0]}")

# ===============================
# CatBoost + Isolation Forest + Azaltılmış Augmentation
# ===============================
X_aug_less, y_aug_less = augment_data(X_iforest_final, y_iforest, noise_level=0.02, multiplier=1)

train_and_evaluate(CatBoostRegressor(verbose=0, random_state=42),
                   "CatBoost with Isolation Forest + LESS Augmentation",
                   X_aug_less, y_aug_less)

print(f"\nReduced Augmentation veri sayısı: {y_aug_less.shape[0]}")

# ===============================
# x6 Feature'ı Çıkarılmış Model
# ===============================

# x6 hariç yeni öznitelik listesi
feature_col_no_x6 = ['x1', 'x2', 'x3', 'x4', 'x5']

# Isolation Forest sonrası veri (x6'sız)
train_df_iforest_no_x6 = isolation_forest_outliers(train_df, feature_col_no_x6)
X_iforest_no_x6 = train_df_iforest_no_x6[feature_col_no_x6]
y_iforest_no_x6 = train_df_iforest_no_x6['Y']

# Non-lineer öznitelikler üret ve SHAP ile en önemli öznitelikleri seç
X_iforest_no_x6_nl = add_non_linear_features(X_iforest_no_x6)
top_features_no_x6 = select_top_features_with_shap(X_iforest_no_x6_nl, y_iforest_no_x6)
X_iforest_no_x6_final = X_iforest_no_x6_nl[top_features_no_x6]

# Modeli eğit ve değerlendir
train_and_evaluate(CatBoostRegressor(verbose=0, random_state=42),
                   "CatBoost with Isolation Forest (x6 Removed)",
                   X_iforest_no_x6_final, y_iforest_no_x6)

print(f"\nIsolation Forest sonrası (x6'sız) veri sayısı: {y_iforest_no_x6.shape[0]}")
# ===============================
# Regülasyonlu CatBoost Karşılaştırması
# ===============================

reg_params = [1, 3, 5, 10, 20]
results = []

for reg in reg_params:
    model = CatBoostRegressor(verbose=0, random_state=42, l2_leaf_reg=reg)
    print(f"\n--- CatBoost with l2_leaf_reg={reg} ---")
    train_and_evaluate(model,
                       f"CatBoost (l2_leaf_reg={reg})",
                       X_iforest_final, y_iforest)
    results.append((reg, model))

# ===============================
# x5 Feature'ı Kullanarak Augmentasyon
# ===============================
def x5_feature_based_augmentation(X, y, noise_level=0.05, multiplier=2):
    augmented_X = [X]
    augmented_y = [y]

    for _ in range(multiplier):
        X_aug = X.copy()
        noise = np.random.normal(0, noise_level, X.shape[0])
        X_aug['x5'] = X_aug['x5'] + noise
        augmented_X.append(X_aug)
        augmented_y.append(y)

    X_total = pd.concat(augmented_X, ignore_index=True)
    y_total = pd.concat(augmented_y, ignore_index=True)

    return X_total, y_total
