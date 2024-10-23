#Q1, 2, 3 Refer This only just change value of x

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'filePath')

x = df['Radio Ad Budget ($)'] #Inplace of Radio Ad Budget Change the parameter which is mentioned
y = df['Sales ($)']

train_size = int(0.8 * len(df))
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_mean = np.mean(x_train)
y_mean = np.mean(y_train)

numerator = np.sum((x_train - x_mean) * (y_train - y_mean))
denominator = np.sum((x_train - x_mean)**2)
m = numerator/denominator
b = y_mean - m * x_mean

print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

y_pred = m * x_test + b

comparison = pd.DataFrame({'Actual Sales': y_test, 'Predicted Sales' : y_pred})
print("\nActual vs Predicted Sales")
print(comparison.head())

rmse = np.sqrt(np.mean((y_test - y_pred)**2))
print(f"\nRoot Mean Squared Error (RMSE): {rmse}")
