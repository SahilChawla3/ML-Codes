import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

data = pd.read_csv("filePath")

x = data.drop(columns=['Id', 'Species'])
y = data['Species']                       

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(x_train, y_train)
y_pred_without_pca = model.predict(x_test)


accuracy_without_pca = accuracy_score(y_test, y_pred_without_pca)
print(f'Accuracy without PCA: {accuracy_without_pca:.2f}')

#PCA start
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(x_pca, y, test_size=0.2, random_state=42)

dt_classifier_pca = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_classifier_pca.fit(x_train_pca, y_train_pca)
y_pred_with_pca = dt_classifier_pca.predict(x_test_pca)

accuracy_with_pca = accuracy_score(y_test_pca, y_pred_with_pca)
print(f'Accuracy with PCA: {accuracy_with_pca:.2f}')

print("\nComparison of Results:")
print(f'Accuracy without PCA: {accuracy_without_pca:.2f}')
print(f'Accuracy with PCA: {accuracy_with_pca:.2f}')
