import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

knndata = pd.read_csv("./knndata.txt")

X = knndata.loc[:, ["X1", "X2"]]
y = knndata.Y

plt.scatter(X[(y == 2)].X1, X[(y == 2)].X2, c="red", marker="+")
plt.scatter(X[(y == 1)].X1, X[(y == 1)].X2, c="blue", marker="o")
plt.show()

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=1)
clf = clf.fit(X, y)
print(clf.predict([[0.7, 0.4]]))
print(clf.predict_proba([[0.7, 0.4]]))

clf = KNeighborsClassifier(n_neighbors=5)
clf = clf.fit(X, y)
print(clf.predict([[0.7, 0.4]]))
print(clf.predict_proba([[0.7, 0.4]]))

clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(X, y)
print(clf.predict([[0.7, 0.6]]))
print(clf.predict_proba([[0.7, 0.6]]))

alldata = pd.read_csv("./alldata.txt")

xtrain = alldata.loc[0:600, ["X1", "X2"]]
ytrain = alldata.loc[0:600, "y"]

xtest = alldata.loc[600:800, ["X1", "X2"]]
ytest = alldata.loc[600:800, "y"]

plt.scatter(xtrain[(ytrain == 2)].X1, xtrain[(ytrain == 2)].X2, c="red", marker="+")
plt.scatter(xtrain[(ytrain == 1)].X1, xtrain[(ytrain == 1)].X2, c="blue", marker="o")
plt.show()

X1 = np.arange (min(xtrain.X1.tolist()), max(xtrain.X1.tolist()), 0.01)
X2 = np.arange (min(xtrain.X2.tolist()), max(xtrain.X2.tolist()), 0.01)
xx, yy = np.meshgrid(X1, X2)

from sklearn import svm
from sklearn.metrics import accuracy_score
clf = svm.SVC(kernel="rbf", gamma=1)
clf = clf.fit(xtrain, ytrain)
pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
pred = pred.reshape(xx.shape)
plt.contour(xx, yy, pred, colors="blue")

clf = svm.SVC(kernel="rbf", gamma=0.01)
clf = clf.fit(xtrain, ytrain)
pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
pred = pred.reshape(xx.shape)
plt.contour(xx, yy, pred, colors="red")

clf = svm.SVC(kernel="rbf", gamma=100)
clf = clf.fit(xtrain, ytrain)
pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
pred = pred.reshape(xx.shape)
plt.contour(xx, yy, pred, colors="green")

plt.scatter(xtrain[(ytrain == 2)].X1, xtrain[(ytrain == 2)].X2, c="red", marker="+")
plt.scatter(xtrain[(ytrain == 1)].X1, xtrain[(ytrain == 1)].X2, c="blue", marker="o")
plt.show()

gammavalues = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]

trainingError = []
testingError = []
for gamma in gammavalues:
    clf = svm.SVC(kernel="rbf", gamma=gamma)
    clf = clf.fit(xtrain, ytrain)
    pred = clf.predict(xtrain)
    trainingError.append(1 - accuracy_score(ytrain, pred))
    pred = clf.predict(xtest)
    testingError.append(1 - accuracy_score(ytest, pred))

plt.plot(trainingError, c="blue")
plt.plot(testingError, c="red")
plt.ylim(0, 0.5)
plt.xticks(range(len(gammavalues)), gammavalues)
plt.legend(["Training Error", "Testing Error"])
plt.xlabel("Gamma")
plt.ylabel("Error")
plt.show()

from sklearn.model_selection import cross_val_score
accuracies = []
for gamma in gammavalues:
    clf = svm.SVC(kernel="rbf", gamma=gamma)
    scores = cross_val_score(clf, xtrain, ytrain, cv=10)
    accuracies.append(scores.mean())

print("Best gamma: ", gammavalues[np.argmax(accuracies)])