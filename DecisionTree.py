from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

iris = pd.read_csv(r'filePath')
x = iris.drop(columns=['Id', 'Species'])
y = iris['Species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model_log_loss = DecisionTreeClassifier(criterion='log_loss', max_depth=4, min_samples_split=2)
model_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_split=2)
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_split=2)

model_log_loss.fit(x_train, y_train)
model_entropy.fit(x_train, y_train)
model_gini.fit(x_train, y_train)

y_pred_test_log_loss = model_log_loss.predict(x_test)
y_pred_test_entropy = model_entropy.predict(x_test)
y_pred_test_gini = model_gini.predict(x_test)

print(f"Log loss: {y_pred_test_log_loss}")
print(f"Entropy: {y_pred_test_entropy}")
print(f"Gini Index: {y_pred_test_gini}")

test_acc_log_loss = accuracy_score(y_test, y_pred_test_log_loss)
test_acc_gini = accuracy_score(y_test, y_pred_test_gini)
test_acc_entropy = accuracy_score(y_test, y_pred_test_entropy)

print(f"Test accuracy of log loss: {test_acc_log_loss*100:.2f}%")
print(f"Test accuracy of gini: {test_acc_gini*100:.2f}%")
print(f"Test accuracy of entropy: {test_acc_entropy*100:.2f}%")