from snowflake_profile import df
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


x = df[['TEMPERATURE','HUMIDITY','RAINFALL','PH','NITROGEN','PHOSPHOROUS','POTASSIUM']]
y = df['SOIL_MOISTURE']


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

lr_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)
dt_predictions = dt_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

print("Linear Regression R2 Score:", r2_score(y_test, lr_predictions))
print("Decision Tree R2 Score:", r2_score(y_test, dt_predictions))
print("Random Forest R2 Score:", r2_score(y_test, rf_predictions))