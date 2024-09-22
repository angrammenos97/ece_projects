import numpy as np

from sklearn import tree
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# import and select data
iris = datasets.load_iris()
data = iris.data[:, [0, 1]]
target = iris.target
target[100:125] = 0
target[125:150] = 1

# split data
xtrain = np.concatenate((data[0:40], data[50:90], data[100:140]))
ytrain = np.concatenate((target[0:40], target[50:90], target[100:140]))
xtest = np.concatenate((data[40:50], data[90:100], data[140:150]))
ytest = np.concatenate((target[40:50], target[90:100], target[140:150]))

# question1 build tree
clf = tree.DecisionTreeClassifier(min_samples_split=20)
clf = clf.fit(xtrain, ytrain)
pred = clf.predict(xtest)

# question2 calculate metrics
print(confusion_matrix(ytest, pred))
print(accuracy_score(ytest, pred))
print(precision_score(ytest, pred, pos_label=1))
print(recall_score(ytest, pred, pos_label=1))
print(f1_score(ytest, pred, pos_label=1))