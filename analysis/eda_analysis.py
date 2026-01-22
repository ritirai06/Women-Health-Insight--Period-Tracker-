import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data_path = os.path.join("..", "/home/ritirai/AI-Powered Women Health Insight System/Women Health Insight/data/women_health_dataset.csv")
df = pd.read_csv(data_path) 
# Display basic info    
print("Dataset Info:")
print(df.head())
print(df.describe())

#stress vs delay_days analysis
plt.scatter(df['stress_level'], df['delay_days'])
plt.title('Stress Level vs Delay Days')
plt.xlabel('Stress Level')
plt.ylabel('Delay Days')
plt.savefig(os.path.join("..", "/home/ritirai/AI-Powered Women Health Insight System/Women Health Insight/PNG OUTPUT/stress_vs_delay_days.png"))
plt.show()

#boxplot for stress vs delay_days
sns.boxplot(x='stress_level', y='delay_days', data=df)
plt.title('Stress Level vs Delay Days (Boxplot)')
plt.savefig(os.path.join("..", "stress_vs_delay_days_boxplot.png"))
plt.show()

corr=df.corr()
print("Correlation Matrix:")
print(corr)
