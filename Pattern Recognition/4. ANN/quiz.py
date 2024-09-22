import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor

X1 = [2,2,-2,-2,1,1,-1,-1]
X2 = [2,-2,-2,2,1,-1,-1,1]
Y = [1,1,1,1,2,2,2,2]
alldata = pd.DataFrame({"X1":X1, "X2":X2, "Y":Y})

xtrain = alldata.loc[:,["X1", "X2"]]
ytrain = alldata.loc[:, "Y"]

clf = MLPClassifier(hidden_layer_sizes=(2), max_iter=10000)
clf.fit(xtrain, ytrain)
pred = clf.predict(xtrain)
trnError = [(t-p) for (t,p) in zip(ytrain,pred)]
AME = np.mean(np.abs(trnError))
print(AME)
##############
clf = MLPRegressor(hidden_layer_sizes=(20), max_iter=10000)
clf.fit(xtrain, ytrain)
pred = clf.predict(xtrain)
trnError = [(t-p) for (t,p) in zip(ytrain,pred)]
AME = np.mean(np.abs(trnError))
print(AME)
#############
clf = MLPRegressor(hidden_layer_sizes=(20,20), max_iter=10000)
clf.fit(xtrain, ytrain)
pred = clf.predict(xtrain)
trnError = [(t-p) for (t,p) in zip(ytrain,pred)]
AME = np.mean(np.abs(trnError))
print(AME)

# clf = MLPRegressor(hidden_layer_sizes=(20,20), max_iter=10000)
# clf = clf.fit(xtrain, ytrain)

# pred = clf.predict(xtrain)
# trainError = [(t-p) for (t,p) in zip(ytrain,pred)]
# AME = np.mean(np.abs(trainError))
# print(AME)