import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

file_path = r"C:\Users\rames\OneDrive\Desktop\my projects\ML1\Sample - Superstore.csv"
df = pd.read_csv(file_path, encoding='latin1')

df['Order Date'] = pd.to_datetime(df['Order Date'])
daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

daily_sales['Year'] = daily_sales['Order Date'].dt.year
daily_sales['Month'] = daily_sales['Order Date'].dt.month
daily_sales['DayOfWeek'] = daily_sales['Order Date'].dt.dayofweek
daily_sales['Quarter'] = daily_sales['Order Date'].dt.quarter

X = daily_sales[['Year', 'Month', 'DayOfWeek', 'Quarter']]
y = daily_sales['Sales']

split_point = int(len(daily_sales) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
dates_test = daily_sales['Order Date'].iloc[split_point:]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: ${mae:.2f}")

plt.figure(figsize=(14, 7))
plt.plot(dates_test, y_test.values, label='Actual Sales', color='blue', alpha=0.6)
plt.plot(dates_test, predictions, label='AI Forecast', color='red', linestyle='--')
plt.title('Superstore Sales Forecast: Actual vs AI Predictions', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Daily Sales ($)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()