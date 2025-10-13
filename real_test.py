import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
data = {
    'Forecasting Horizon': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    'Highest R² Score': [0.3490, 0.3681, 0.3754, 0.3673, 0.3618, 0.3486, 0.3393, 0.3249, 0.3173, 0.3042, 0.2938, 0.2793, 0.2688, 0.2595, 0.2544, 0.2437, 0.2362, 0.2252, 0.2196, 0.2171, 0.2182, 0.2018, 0.1885, 0.1871, 0.2362, 0.1785, 0.1698, 0.1599, 0.1615, 0.1615]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Descriptive Statistics
print("Descriptive Statistics:")
print(df.describe())

# Check for anomalies: Horizon 26 has an unusually high R² score compared to its neighbors
horizon_26 = df[df['Forecasting Horizon'] == 26]
print("\nAnomaly at Horizon 26:")
print(horizon_26)

# Calculate the correlation of R² score with Forecasting Horizon
correlation = df['Forecasting Horizon'].corr(df['Highest R² Score'])
print(f"\nCorrelation between Forecasting Horizon and R² Score: {correlation:.4f}")

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(df['Forecasting Horizon'], df['Highest R² Score'], marker='o', color='b', linestyle='-', markersize=6)
plt.title('Highest R² Score vs. Forecasting Horizon')
plt.xlabel('Forecasting Horizon')
plt.ylabel('Highest R² Score')
plt.grid(True)
plt.xticks(df['Forecasting Horizon'])  # Ensure all horizons are shown
plt.tight_layout()

# Show plot
plt.show()