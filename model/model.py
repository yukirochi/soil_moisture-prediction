from snowflake_profile import df
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

df = df.dropna()
x = np.array(df[['ATMOSPHERIC_TEMP','HUMIDITY','SOIL_TEMP','RAINFALL','PREVIOUS_SOIL_MOISTURE','SOIL_MOISTURE']]).reshape(-1, 6)
y = np.array(df['FUTURE_SOIL_MOISTURE']).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)


print("Linear Regression R2 Score:", r2_score(y_test, lr_predictions))
print("Linear Regression MSE:", mean_squared_error(y_test, lr_predictions))

# # 1. Zoomed-in Time Series Plot (2 weeks / 336 hours)
# subset_size = 336 # 24 hours * 14 days

# plt.figure(figsize=(14, 6))
# # Plotting just the first 336 hours of the test set
# plt.plot(y_test[:subset_size], label='Actual Moisture', color='blue', alpha=0.8, linewidth=2)
# plt.plot(lr_predictions[:subset_size], label='Predicted Moisture', color='red', alpha=0.8, linestyle='--', linewidth=2)

# plt.title('Time Series Detail: 2-Week Hourly Window')
# plt.xlabel('Time (Hours)')
# plt.ylabel('Soil Moisture Level')
# plt.legend()
# plt.grid(True, alpha=0.3)

# plt.savefig('zoomed_timeseries.png', dpi=300, bbox_inches='tight')
# print("Saved zoomed time series plot!")



