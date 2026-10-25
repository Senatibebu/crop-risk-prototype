import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print("Loading data...")
data = pd.read_csv('data.csv')

X = data[['soil_moisture', 'temperature']]
y = data['yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the model...")
model = LinearRegression()
model.fit(X_train, y_train)

new_sensor_data = [[35, 24]] 
prediction = model.predict(new_sensor_data)

print(f"Predicted Yield for Soil Moisture 35 and Temp 24 is: {prediction[0]:.2f} tons/hectare")