import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor, MLPClassifier

# anndata = pd.read_csv("lab.txt")
# X = anndata.loc[:, ["X1", "X2"]]
# y = anndata.Y

# clf = MLPRegressor(hidden_layer_sizes=(), learning_rate_init=1)
# clf = clf.fit(X, y)

# print(clf.predict([[0, 0]]))
# print(clf.predict([[1, 0]]))
# print(clf.predict([[0.5, 0]]))

# plt.scatter(anndata[(anndata.Y==-1)].X1, anndata[anndata.Y==-1].X2, c='red', marker='+')
# plt.scatter(anndata[(anndata.Y== 1)].X1, anndata[anndata.Y== 1].X2, c='blue', marker='o')
# plt.show()

alldata = pd.read_csv("alldata.txt")
xtrain = alldata.loc[0:600, ["X1", "X2"]]
ytrain = alldata.loc[0:600, "y"]
xtest = alldata.loc[600:, ["X1", "X2"]]
ytest = alldata.loc[600:, "y"]

clf = MLPRegressor(hidden_layer_sizes=(20,20), tol=0.01)
clf = clf.fit(xtrain, ytrain)

pred = clf.predict(xtrain)
trainError = [(t-p) for (t,p) in zip(ytrain, pred)]
MAE = np.mean(np.abs(trainError))
print(MAE)
plt.hist(trainError, range=(-1,1), rwidth=0.5)
plt.show()

pred = clf.predict(xtest)
testingError = [(t - p) for (t, p) in zip(ytest, pred)]
MAE = np.mean(np.abs(testingError))
print(MAE)
plt.hist(testingError, range=(-1, 1), rwidth=0.5)
plt.show()

# plt.scatter(xtrain[ytrain==2].X1, xtrain[ytrain==2].X2, c='red', marker='+')
# plt.scatter(xtrain[ytrain==1].X1, xtrain[ytrain==1].X2, c='blue', marker='o')
# plt.show()