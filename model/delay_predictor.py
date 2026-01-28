import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error 
import joblib

print("=" * 80)
print("TRAINING ENHANCED MENSTRUAL DELAY PREDICTION MODEL")
print("=" * 80)

# Load dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "women_health_dataset.csv")
df = pd.read_csv(data_path)

print(f"\n📊 Dataset loaded successfully!")
print(f"   Total samples: {len(df)}")
print(f"   Features: {len(df.columns)}")
print(f"   Hormonal imbalance cases: {df['hormonal_imbalance'].sum()} ({df['hormonal_imbalance'].sum()/len(df)*100:.1f}%)")

# Drop user_id as it's not a feature
if 'user_id' in df.columns:
    df = df.drop('user_id', axis=1)

# Encode categorical variables
print(f"\n🔄 Encoding categorical variables...")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"   Categorical columns: {categorical_cols}")

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print(f"   Encoded features: {len(df_encoded.columns) - 1}")

# Separate features and target
X = df_encoded.drop('delay_days', axis=1)
y = df_encoded['delay_days']

print(f"\n📋 Feature names:")
for i, col in enumerate(X.columns, 1):
    print(f"   {i}. {col}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n🔀 Data split:")
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# Train model
print(f"\n🤖 Training Random Forest Regressor...")
print(f"   Estimators: 100 trees")
print(f"   Random state: 42")

model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=20, min_samples_split=5)
model.fit(X_train, y_train)

print(f"   ✅ Model training completed!")

# Make predictions
print(f"\n🎯 Making predictions...")
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"\n" + "=" * 80)
print("MODEL PERFORMANCE METRICS")
print("=" * 80)
print(f"Mean Absolute Error (MAE):        {mae:.2f} days")
print(f"Mean Squared Error (MSE):         {mse:.2f}")
print(f"Root Mean Squared Error (RMSE):   {rmse:.2f} days")
print("=" * 80)

# Feature importance
print(f"\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)
print(f"{'Rank':<6} {'Feature':<35} {'Importance':<12} {'%':<8}")
print("-" * 80)

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

for idx, row in feature_importance_df.head(15).iterrows():
    rank = feature_importance_df.index.get_loc(idx) + 1
    percentage = row['importance'] * 100
    bar = '█' * int(percentage * 2)
    print(f"{rank:<6} {row['feature']:<35} {row['importance']:<12.4f} {percentage:>6.2f}% {bar}")

print("=" * 80)

# Save the model
model_path = os.path.join(BASE_DIR, "model", "delay_predictor_model.pkl")
print(f"\n💾 Saving model to: {model_path}")
joblib.dump(model, model_path)
print(f"   ✅ Model saved successfully!")

# Test predictions on different scenarios
print(f"\n" + "=" * 80)
print("SAMPLE PREDICTIONS")
print("=" * 80)

# Test on cases with hormonal imbalance
hormonal_cases = df_encoded[df_encoded['hormonal_imbalance'] == 1].head(3)
normal_cases = df_encoded[df_encoded['hormonal_imbalance'] == 0].head(3)

print("\n🔴 Hormonal Imbalance Cases:")
for idx in hormonal_cases.index:
    actual = hormonal_cases.loc[idx, 'delay_days']
    features = hormonal_cases.drop('delay_days', axis=1).loc[idx:idx]
    predicted = model.predict(features)[0]
    print(f"   Case {idx}: Actual = {actual:.0f} days, Predicted = {predicted:.0f} days")

print("\n🟢 Normal Cases:")
for idx in normal_cases.index:
    actual = normal_cases.loc[idx, 'delay_days']
    features = normal_cases.drop('delay_days', axis=1).loc[idx:idx]
    predicted = model.predict(features)[0]
    print(f"   Case {idx}: Actual = {actual:.0f} days, Predicted = {predicted:.0f} days")

print("\n" + "=" * 80)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 80)  