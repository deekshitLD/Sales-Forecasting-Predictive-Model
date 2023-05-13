import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the sales data
sales_data = pd.read_csv('sales_data.csv')

# Preprocess the data
# - Convert the date column to datetime format
sales_data['date'] = pd.to_datetime(sales_data['date'])

# - Create time-based features such as month, day, and year
sales_data['month'] = sales_data['date'].dt.month
sales_data['day'] = sales_data['date'].dt.day
sales_data['year'] = sales_data['date'].dt.year

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sales_data[['month', 'day', 'year']], sales_data['sales'], test_size=0.2, random_state=42)

# Train a random forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
