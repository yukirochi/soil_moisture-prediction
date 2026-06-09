from snowflake_profile import df
import numpy as np
from sklearn.model_selection import train_test_split,TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from sklearn.utils import shuffle


df = df.dropna()
x = np.array(df[['ATMOSPHERIC_TEMP','HUMIDITY','SOIL_TEMP','RAINFALL','PREVIOUS_SOIL_MOISTURE','SOIL_MOISTURE']]).reshape(-1, 6)
y = np.array(df['FUTURE_SOIL_MOISTURE']).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
real_score = r2_score(y_test, lr_model.predict(X_test))


y_shuffled = shuffle(y_train, random_state=42)
lr_model_shuffled = LinearRegression()
lr_model_shuffled.fit(X_train, y_shuffled)
shuffled_score = r2_score(y_test, lr_model_shuffled.predict(X_test))

print(f"Real target R2:     {real_score:.4f}")
print(f"Shuffled target R2: {shuffled_score:.4f}")

#result 
# Real target R2:     0.9145
# Shuffled target R2: -1596.2960