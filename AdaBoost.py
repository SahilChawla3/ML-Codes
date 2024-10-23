# Same for Bagging, decision tree, random_forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

iris = pd.read_csv(r'filePath')
x = iris.drop(columns=['Id', 'Species'])
y = iris['Species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = AdaBoostClassifier(n_estimators=102, random_state=42)
model.fit(x_train, y_train)

y_pred_test = model.predict(x_test)

print(y_pred_test)

test_acc = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")