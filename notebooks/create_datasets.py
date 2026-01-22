import pandas as pd
import numpy as np
#create dummy datatset
data = {
    "cycle_length": np.random.randint(25,45,500),
    "period_duration": np.random.randint(2,7,500),
    "flow_level": np.random.choice(['light', 'medium', 'heavy'],500),
    "stress_level": np.random.choice(['low', 'medium', 'high'],500),
    "sleep_hours": np.random.uniform(4,9,500),
    "delay_days": np.random.randint(0,10,500)
}   
df=pd.DataFrame(data)
df.to_csv("/home/ritirai/AI-Powered Women Health Insight System/Women Health Insight/data/women_health_dataset.csv", index=False)
print("Dummy dataset created and saved to /home/ritirai/AI-Powered Women Health Insight System/Women Health Insight/data/women_health_dataset.csv")   