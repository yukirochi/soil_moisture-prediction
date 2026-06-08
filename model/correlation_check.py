from snowflake_profile import df
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CORRELATION CHECK
# ---------------------------------------------------------

# 1. Print the raw math to the terminal
print("\n--- Correlation to Soil Moisture ---")
# This calculates how strongly every column relates to SOIL_MOISTURE specifically
correlations = df.corr()['SOIL_MOISTURE'].sort_values(ascending=False)
print(correlations)
print("------------------------------------\n")

# 2. Generate a visual Heatmap
plt.figure(figsize=(10, 8))
# annot=True puts the numbers inside the squares, cmap adds color logic
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title("Dataset Correlation Heatmap")
plt.tight_layout()
plt.show()