from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('exemplo2.csv')

X = df.drop('risco', axis=1)
y = df.risco

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2/3)
print(X_test.shape)
print(y_test.shape)

knn2 = KNeighborsClassifier(n_neighbors=3)

knn2.fit(X_train, y_train)

print(accuracy_score(y_test, knn2.predict(X_test)))