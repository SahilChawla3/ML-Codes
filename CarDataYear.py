import numpy as np
import pandas as pd

df = pd.read_csv(r'filePath')

# print(df.info())

df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Selling_Price'] = pd.to_numeric(df['Selling_Price'], errors='coerce')

# print(df['Year'].isnull().sum())

df.dropna(subset=['Year', 'Selling_Price'], inplace=True)
x = df['Year'].to_numpy()
y = df['Selling_Price'].to_numpy()

train_size = int(0.8 * len(df))
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

x_mean = np.mean(x_train)
y_mean = np.mean(y_train)

numerator = np.sum((x_train - x_mean) * (y_train - y_mean))
denominator = np.sum((x_train - x_mean)**2)
m = numerator/denominator
b = y_mean - m * x_mean

print(f"\nSlope (m): {m}")
print(f"Intercept (b): {b}")

y_pred = m * x_test + b

comparison = pd.DataFrame({'Actual Selling Price': y_test, 'Predicted Selling Price': y_pred})
print("\nActual vs Predicted Selling Prices:")
print(comparison.head())

rmse = np.sqrt(np.mean(y_test - y_pred)**2)
print(rmse)