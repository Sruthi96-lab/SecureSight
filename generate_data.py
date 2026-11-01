
import os
import numpy as np
import pandas as pd

# Reproducible demo data
np.random.seed(42)

os.makedirs("data", exist_ok=True)

normal_users = 30
events_per_user = 20
total_events = normal_users * events_per_user

data = []

for i in range(total_events):
    hour = int(np.random.choice(range(8, 19)))
    login_failures = int(np.random.poisson(0.3))
    files_accessed = int(np.random.randint(1, 16))
    data_transfer_mb = round(float(np.random.exponential(15)), 2)
    usb_transfer = int(np.random.choice([0, 1], p=[0.95, 0.05]))
    off_hours = int(hour < 8 or hour >= 19)

    data.append({
        "event_id": i + 1,
        "user_id": f"user_{(i % normal_users) + 1:02d}",
        "hour": hour,
        "login_failures": login_failures,
        "files_accessed": files_accessed,
        "data_transfer_mb": data_transfer_mb,
        "usb_transfer": usb_transfer,
        "off_hours": off_hours,
        "data_source": "synthetic_demo"
    })

# Add unusual activity for testing
for i in range(15):
    data.append({
        "event_id": total_events + i + 1,
        "user_id": f"user_{np.random.randint(1, normal_users + 1):02d}",
        "hour": int(np.random.choice([0, 1, 2, 3, 23])),
        "login_failures": int(np.random.randint(5, 12)),
        "files_accessed": int(np.random.randint(40, 101)),
        "data_transfer_mb": round(
            float(np.random.uniform(500, 2000)), 2
        ),
        "usb_transfer": 1,
        "off_hours": 1,
        "data_source": "synthetic_demo"
    })

df = pd.DataFrame(data)
df.to_csv("data/user_activity.csv", index=False)

print("Dataset generated successfully!")
print(f"Total activity records: {len(df)}")
print("File saved to: data/user_activity.csv")
print("\nFirst five records:")
print(df.head())
