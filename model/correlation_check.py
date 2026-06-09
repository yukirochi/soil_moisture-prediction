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

# --- Correlation to Soil Moisture ---
# SOIL_MOISTURE             1.000000
# PREVIOUS_SOIL_MOISTURE    0.999829
# FUTURE_SOIL_MOISTURE      0.999829
# RAINFALL                  0.092532
# HUMIDITY                 -0.149249
# ATMOSPHERIC_TEMP         -0.300211
# SOIL_TEMP                -0.582926
# Name: SOIL_MOISTURE, dtype: float64
# ------------------------------------