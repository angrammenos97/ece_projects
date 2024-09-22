import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

X1 = [-2.0,-2.0,-1.8,-1.4,-1.2,1.2,1.3,1.3,2.0,2.0,-0.9,-0.5,-0.2,0.0,0.0,0.3,0.4,0.5,0.8,1.0]
X2 = [-2.0,1.0,-1.0,2.0,1.2,1.0,-1.0,2.0,0.0,-2.0,0.0,-1.0,1.5,0.0,-0.5,1.0,0.0,-1.5,1.5,0.0]
Y = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2]
alldata = pd.DataFrame({"X1":X1, "X2":X2, "Y":Y})
X = alldata.loc[:,["X1","X2"]]
y = alldata.Y

# plt.scatter(X[y==1].X1, X[y==1].X2, c='red', marker='+')
# plt.scatter(X[y==2].X1, X[y==2].X2, c='blue', marker='o')
# plt.show()
#####################
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X,y)
print(clf.predict([[1.5,-0.5]]))
#####################
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X,y)
print(clf.predict_proba([[-1,1]]))
# # # # # # # # # # # # # # # # # # # 
X1 = [2,2,-2,-2,1,1,-1,-1]
X2 = [2,-2,-2,2,1,-1,-1,1]
Y = [1,1,1,1,2,2,2,2]
alldata = pd.DataFrame({"X1":X1, "X2":X2, "Y":Y})
X = alldata.loc[:,["X1","X2"]]
y = alldata.Y
X1 = np.arange(min(X.X1.tolist()), max(X.X1.tolist()), 0.01)
X2 = np.arange(min(X.X2.tolist()), max(X.X2.tolist()), 0.01)
xx, yy = np.meshgrid(X1, X2)

clf = svm.SVC(kernel='rbf', gamma=1)
clf.fit(X,y)
pred = clf.predict(X)
print(accuracy_score(y, pred))
########################
clf = svm.SVC(kernel='rbf', gamma=1000000)
clf.fit(X,y)
print(clf.predict([[-2,-1.9]]))
########################
plt.scatter(X[y==1].X1, X[y==1].X2, c="red", marker="+")
plt.scatter(X[y==2].X1, X[y==2].X2, c="blue", marker="o")
points = [-2,-1.9]
plt.scatter(points[0], points[1], c="green")
pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
pred = pred.reshape(xx.shape)
plt.contour(xx, yy, pred, colors='orange')
plt.show()

# clf = KNeighborsClassifier(n_neighbors=3)
# clf = clf.fit(X, y)
# print(clf.predict([[1.5,-0.5]]))

# clf = KNeighborsClassifier(n_neighbors=5)
# clf = clf.fit(X, y)
# print(clf.predict_proba([[-1,1]]))

# plt.scatter(X[y==1].X1, X[y==1].X2, c="red", marker="+")
# plt.scatter(X[y==2].X1, X[y==2].X2, c="blue", marker="o")
# plt.show()

# X1 = [2,2,-2,-2,1,1,-1,-1]
# X2 = [2,-2,-2,2,1,-1,-1,1]
# Y = [1,1,1,1,2,2,2,2]
# alldata = pd.DataFrame({"X1":X1, "X2":X2, "Y":Y})
# X = alldata.loc[:,["X1","X2"]]
# y = alldata.Y
# X1 = np.arange(min(X.X1.tolist()), max(X.X1.tolist()), 0.01)
# X2 = np.arange(min(X.X2.tolist()), max(X.X2.tolist()), 0.01)
# xx, yy = np.meshgrid(X1, X2)

# clf = svm.SVC(kernel='rbf', gamma=1)
# clf = clf.fit(X, y)
# pred = clf.predict(X)
# print(accuracy_score(y, pred))

# pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# pred = pred.reshape(xx.shape)
# plt.contour(xx, yy, pred, colors='yellow')

# clf = svm.SVC(kernel='rbf', gamma=1000000)
# clf = clf.fit(X, y)
# points = [-2,-1.9]
# print(clf.predict([points]))

# pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# pred = pred.reshape(xx.shape)
# plt.contour(xx, yy, pred, colors='orange')

# plt.scatter(X[y==1].X1, X[y==1].X2, c="red", marker="+")
# plt.scatter(X[y==2].X1, X[y==2].X2, c="blue", marker="o")
# plt.scatter(points[0], points[1], c="green")
# plt.show()