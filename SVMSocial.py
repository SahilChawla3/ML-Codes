import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r'filePath')

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

x = df[['Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = SVC(kernel='rbf', random_state=42) #change linear for linear kernel
model.fit(x_train, y_train)

y_pred_test = model.predict(x_test)
print(y_pred_test)

test_acc = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
conf_matrix = confusion_matrix(y_test, y_pred_test)

print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)

y_pred_output = ['Yes' if pred == 1 else 'No' for pred in y_pred_test]
print("\nPrediction for whether a user purchased a car:")
print(y_pred_output)