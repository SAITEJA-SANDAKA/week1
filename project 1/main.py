import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load your dataset (assuming it’s in CSV format)
data = pd.read_csv("C:\\Users\\150LAB\\Desktop\\03f4d1c1a55947025601.csv")


# Step 2: Preprocess data
# Assuming the columns are: 'irradiance', 'temperature', 'humidity', 'wind_speed', 'solar_power_output'
x=data.columns[1:]
X = data[list(x)]  # Features
y = data[data.columns[0]]  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize the linear regression model
model = LinearRegression()

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Predict the solar power output on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

# Step 7: Plot the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Solar Power Output")
plt.ylabel("Predicted Solar Power Output")
plt.title("Actual vs Predicted Solar Power Output")
plt.show()
