import pandas as pd
import numpy as np

np.random.seed(42)

# Define ranges
dates = pd.date_range(start="2023-01-01", end="2023-06-30", freq="W")
stores = [f"Store_{i}" for i in range(1, 6)]
departments = [f"Dept_{i}" for i in range(1, 8)]
regions = ["North", "South", "East", "West"]

# Generate rows
data = []
for date in dates:
    for store in stores:
        for dept in np.random.choice(departments, size=3, replace=False):
            region = np.random.choice(regions)
            weekly_sales = round(np.random.uniform(2000, 15000), 2)
            transactions = int(weekly_sales / np.random.uniform(20, 80))
            data.append([date, store, dept, region, weekly_sales, transactions])

df = pd.DataFrame(
    data,
    columns=["date", "store", "department", "region", "weekly_sales", "transactions"]
)

# Add a few intentional nulls and anomalies for testing
df.loc[np.random.choice(df.index, 10), "weekly_sales"] = np.nan
df.loc[np.random.choice(df.index, 5), "transactions"] = np.nan
df.loc[np.random.choice(df.index, 3), "weekly_sales"] = df["weekly_sales"].max() * 3  # outliers

# Save
df.to_csv("sample_data/sales_data.csv", index=False)
print("✅ Sample dataset created: sample_data/sales_data.csv")
print(df.head())
