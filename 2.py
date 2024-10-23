#Q4, 5, 6
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'filePath')

x = df[['Radio Ad Budget ($)', 'TV Ad Budget ($)']]
y = df['Sales ($)']

train_size = int(0.8 * len(df))
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_mean = np.mean(x_train, axis=0)
y_mean = np.mean(y_train)

numerator = np.sum((x_train - x_mean) * (y_train[:, np.newaxis] - y_mean), axis=0)
denominator = np.sum((x_train - x_mean)**2, axis = 0)

m_radio = numerator[0] / denominator[0]
m_tv = numerator[1] / denominator[1]
b = y_mean - m_radio * x_mean[0] - m_tv * x_mean[1]

print(f"Slope for Radio (m_radio): {m_radio}")
print(f"Slope for TV (m_tv): {m_tv}")
print(f"Intercept (b): {b}")

y_pred = m_radio * x_test[:, 0] + m_tv * x_test[:, 1] + b

comparison = pd.DataFrame({'Actual Sales':y_test, 'Predicted Sales':y_pred})
print("\nActual vs Predicted Sales:")
print(comparison.head())

rmse = np.sqrt(np.mean((y_test - y_pred)**2))
print(f"\nRoot Mean Squared Error (RMSE): {rmse}")

'''
Explanation of X_test[:, 0]:
X_test is a 2D NumPy array (matrix) where each row is a data point, and each column corresponds to a feature.
[:, 0] means "select all rows (:) from the first column (0)". In this case, the first column represents the Radio budget.
[:, 1] means "select all rows from the second column", which represents the TV budget.
So, when we write X_test[:, 0], it extracts the Radio budget for all test samples, and X_test[:, 1] extracts the TV budget for all test samples.


-----------------------------------------------------------------------------------------------------------------


Explanation of axis=0 in X_mean = np.mean(X_train, axis=0):
X_train is a 2D array where each row represents a data point, and each column represents a feature (Radio and TV).
axis=0 means "compute the mean along the columns" (i.e., for each feature across all data points).


-----------------------------------------------------------------------------------------------------------------


y[:, np.newaxis] converts a 1D array into a 2D column array, making it compatible for operations with multi-dimensional arrays.
It's used to align the dimensions of y with other arrays (like X_train) when performing element-wise or matrix operations.
'''
