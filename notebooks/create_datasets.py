import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

# -----------------------------
# Demographics
# -----------------------------
age = np.random.randint(15, 50, n_samples)

# -----------------------------
# Medical conditions
# -----------------------------
has_pcos = np.random.binomial(1, 0.15, n_samples)
has_thyroid = np.random.binomial(1, 0.12, n_samples)
has_endometriosis = np.random.binomial(1, 0.10, n_samples)

# -----------------------------
# Lifestyle
# -----------------------------
stress_level = np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.3, 0.45, 0.25])
sleep_hours = np.round(np.random.uniform(4, 9, n_samples), 1)
exercise_frequency = np.random.choice(
    ['sedentary', 'light', 'moderate', 'active'],
    n_samples, p=[0.25, 0.30, 0.30, 0.15]
)
diet_quality = np.random.choice(
    ['poor', 'fair', 'good', 'excellent'],
    n_samples, p=[0.15, 0.30, 0.40, 0.15]
)
water_intake = np.random.randint(2, 12, n_samples)

# -----------------------------
# Body metrics
# -----------------------------
height = np.random.uniform(150, 180, n_samples)
weight = np.random.uniform(45, 90, n_samples)
bmi = np.round(weight / ((height / 100) ** 2), 2)

# -----------------------------
# Hormonal imbalance (bounded)
# -----------------------------
hormonal_imbalance = np.zeros(n_samples, dtype=int)

for i in range(n_samples):
    risk = 0.0
    risk += 0.35 if has_pcos[i] else 0
    risk += 0.25 if has_thyroid[i] else 0
    risk += 0.15 if stress_level[i] == 'high' else 0
    risk += 0.15 if sleep_hours[i] < 6 else 0
    risk += 0.15 if bmi[i] < 18.5 or bmi[i] > 30 else 0

    risk = min(risk, 0.9)
    hormonal_imbalance[i] = np.random.rand() < risk

# -----------------------------
# Cycle length (realistic)
# -----------------------------
cycle_length = np.zeros(n_samples, dtype=int)

for i in range(n_samples):
    if hormonal_imbalance[i] == 0:
        cycle_length[i] = np.random.randint(24, 36)
    else:
        if np.random.rand() < 0.25:
            cycle_length[i] = np.random.randint(90, 151)  # skipped cycles
        else:
            cycle_length[i] = np.random.randint(36, 75)   # irregular cycles

# -----------------------------
# Delay derived from cycle
# -----------------------------
delay_days = np.maximum(0, cycle_length - 28)

# -----------------------------
# Period characteristics
# -----------------------------
period_duration = np.random.randint(3, 8, n_samples)
flow_level = np.random.choice(['light', 'medium', 'heavy'], n_samples, p=[0.25, 0.5, 0.25])
cramp_severity = np.random.randint(0, 11, n_samples)

# -----------------------------
# Contraceptive (age-aware)
# -----------------------------
contraceptive_use = []
for a in age:
    if a < 18:
        contraceptive_use.append('none')
    else:
        contraceptive_use.append(
            np.random.choice(
                ['none', 'oral contraceptive', 'IUD', 'implant'],
                p=[0.45, 0.30, 0.15, 0.10]
            )
        )

# -----------------------------
# Mood
# -----------------------------
mood_state = np.random.choice(
    ['excellent', 'good', 'neutral', 'anxious', 'depressed'],
    n_samples, p=[0.15, 0.35, 0.30, 0.15, 0.05]
)

# -----------------------------
# Final Dataset
# -----------------------------
df = pd.DataFrame({
    "user_id": range(1, n_samples + 1),
    "age": age,
    "cycle_length": cycle_length,
    "period_duration": period_duration,
    "flow_level": flow_level,
    "stress_level": stress_level,
    "sleep_hours": sleep_hours,
    "exercise_frequency": exercise_frequency,
    "water_intake": water_intake,
    "diet_quality": diet_quality,
    "weight": np.round(weight, 1),
    "height": np.round(height, 1),
    "bmi": bmi,
    "contraceptive_use": contraceptive_use,
    "has_pcos": has_pcos,
    "has_endometriosis": has_endometriosis,
    "has_thyroid": has_thyroid,
    "mood_state": mood_state,
    "cramp_severity": cramp_severity,
    "hormonal_imbalance": hormonal_imbalance,
    "delay_days": delay_days
})

# SAME DATASET NAME (UNCHANGED)
output_path = "/home/ritirai/Desktop/RITI/AI-Powered Women Health Insight System/Women Health Insight/data/women_health_dataset.csv"
df.to_csv(output_path, index=False)

print("Dataset generated successfully with realistic constraints.")
print(f"Saved at: {output_path}")
