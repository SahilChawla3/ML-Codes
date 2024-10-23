from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

iris = pd.read_csv(r'filePath')
x = iris.drop(columns=['Id', 'Species'])
y = iris['Species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear', random_state=42)
model.fit(x_train, y_train)

y_pred_test = model.predict(x_test)
print(y_pred_test)

test_acc = accuracy_score(y_test, y_pred_test)
test_prec = precision_score(y_test, y_pred_test, average='macro')
test_recall = recall_score(y_test, y_pred_test, average='macro')
test_f1 = f1_score(y_test, y_pred_test, average='macro')
test_confusion_matrix = confusion_matrix(y_test, y_pred_test)

print(f"Test accuracy: {test_acc * 100:.2f}%")
print(f"Test precision: {test_prec * 100:.2f}%")
print(f"Test recall: {test_recall * 100:.2f}%")
print(f"Test f1: {test_f1 * 100:.2f}%")
print("\nConfusion Matrix")
print(test_confusion_matrix)