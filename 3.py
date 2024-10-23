#Just change score parameter i.e. accuracy, precision, recall, f1, etc.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r'filePath')

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

x = df[['Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear', random_state=42)
model.fit(x_train, y_train)

y_pred_test = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred_test)
conf_matrix = confusion_matrix(y_test, y_pred_test)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)

# y_pred_output = ['Yes' if pred == 1 else 'No' for pred in y_pred_test]
# print("\nPrediction for whether a user purchased a car:")
# print(y_pred_output)