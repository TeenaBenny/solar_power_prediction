
# ============================================================
# SOLAR POWER GENERATION PREDICTION
# Dataset: Your own 200-row CSV file
# Target: Power_Output_kWh
# Model: Random Forest Regressor
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. LOAD YOUR DATASET

df = pd.read_csv("solar_power_dataset.csv")

print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# 2. EXPLORE THE DATASET (EDA)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# 3. DATA CLEANING AND DATE PROCESSING

# Convert Date into datetime format
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Convert numeric columns to numeric format
numeric_columns = [
    "Temperature_C",
    "Humidity_pct",
    "Wind_Speed_kmh",
    "Solar_Irradiance_Wm2",
    "Cloud_Cover_pct",
    "Panel_Tilt_deg",
    "Power_Output_kWh"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Target must be present for supervised learning
df = df.dropna(subset=["Power_Output_kWh", "Date"])

# Remove duplicate records
df = df.drop_duplicates().copy()

# Create useful time features from Date
df["Month"] = df["Date"].dt.month
df["DayOfYear"] = df["Date"].dt.dayofyear

print("\nCleaned dataset shape:", df.shape)

# 4. EXPLORATORY DATA VISUALIZATION

# Graph 1: Distribution of daily solar energy
plt.figure(figsize=(8, 5))
sns.histplot(df["Power_Output_kWh"], kde=True)
plt.title("Distribution of Solar Power Output")
plt.xlabel("Daily Energy Generated (kWh)")
plt.tight_layout()
plt.show()

# Graph 2: Solar irradiance vs power output
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Solar_Irradiance_Wm2",
    y="Power_Output_kWh"
)
plt.title("Solar Irradiance vs Power Output")
plt.tight_layout()
plt.show()

# Graph 3: Temperature vs power output
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Temperature_C",
    y="Power_Output_kWh"
)
plt.title("Temperature vs Power Output")
plt.tight_layout()
plt.show()

# Graph 4: Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(
    df[numeric_columns + ["Month", "DayOfYear"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 5. SELECT INPUT FEATURES AND TARGET

features = [
    "Temperature_C",
    "Humidity_pct",
    "Wind_Speed_kmh",
    "Solar_Irradiance_Wm2",
    "Cloud_Cover_pct",
    "Panel_Tilt_deg",
    "Month",
    "DayOfYear"
]

target = "Power_Output_kWh"

X = df[features]
y = df[target]

# 6. SPLIT THE DATASET
# Random split is suitable for this basic demonstration.
# For future forecasting, use a chronological split instead.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 7. PREPROCESSING AND MODEL TRAINING
# Median imputation handles missing input feature values.

model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )
)

model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# 8. MODEL EVALUATION

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION RESULTS")
print("-------------------------")
print(f"MAE  : {mae:.2f} kWh")
print(f"RMSE : {rmse:.2f} kWh")
print(f"R2   : {r2:.3f}")

# 9. ACTUAL VS PREDICTED GRAPH

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)

low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())

plt.plot([low, high], [low, high], "r--")
plt.xlabel("Actual Power Output (kWh)")
plt.ylabel("Predicted Power Output (kWh)")
plt.title("Actual vs Predicted Solar Energy")
plt.tight_layout()
plt.show()

# 10. FEATURE IMPORTANCE

rf_model = model.named_steps["randomforestregressor"]

importance = pd.Series(
    rf_model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature importance:")
print(importance)

importance.plot(kind="bar", figsize=(9, 5))
plt.title("Factors Affecting Solar Power Predictions")
plt.ylabel("Relative Importance")
plt.tight_layout()
plt.show()

# 11. PREDICT FOR NEW INPUT CONDITIONS
# Replace these example values with realistic values
# based on your own dataset.

new_data = pd.DataFrame([{
    "Temperature_C": 30,
    "Humidity_pct": 60,
    "Wind_Speed_kmh": 10,
    "Solar_Irradiance_Wm2": 800,
    "Cloud_Cover_pct": 20,
    "Panel_Tilt_deg": 25,
    "Month": 5,
    "DayOfYear": 125
}])

prediction = model.predict(new_data)[0]

print("\nNEW SOLAR POWER PREDICTION")
print("--------------------------")
print(f"Predicted daily energy: {prediction:.2f} kWh")

print("\nProject completed successfully!")


#Save your trained model

import joblib

joblib.dump(model, "solar_model.pkl")

print("Model saved successfully!")
