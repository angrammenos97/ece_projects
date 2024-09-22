import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier


# question1 import data
knndata = pd.read_csv("./knndata.txt")

# question2 apply kNN
X = knndata.loc[:, ["X1", "X2"]]
y = knndata.Y
plt.scatter(X[(y == 2)].X1, X[(y == 2)].X2, c="red", marker="+")
plt.scatter(X[(y == 1)].X1, X[(y == 1)].X2, c="blue", marker="o")
plt.show()
clf = KNeighborsClassifier(n_neighbors=1)
clf = clf.fit(X, y)
print(clf.predict([[0.7, 0.4]]))
print(clf.predict_proba([[0.7, 0.4]]))
