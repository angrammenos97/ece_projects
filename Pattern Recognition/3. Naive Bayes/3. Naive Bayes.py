import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import CategoricalNB

traffic = pd.read_csv("./traffic.txt")
X = traffic.loc[:, ["Weather", "Day"]]
y = traffic.loc[:, "HighTraffic"]

encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
encoder = encoder.fit(X)
X = encoder.transform(X)

clf = CategoricalNB(alpha=0)
clf.fit(X, y)

new_data = pd.DataFrame({"Weather": ["Hot"], "Day": ["Vacation"]})
transformed_new_data = encoder.transform(new_data)
print(clf.predict(transformed_new_data))
print(clf.predict_proba(transformed_new_data))

clf = CategoricalNB(alpha=1)
clf.fit(X, y)

new_data = pd.DataFrame({"Weather": ["Hot"], "Day": ["Weekend"]})
transformed_new_data = encoder.transform(new_data)
print(clf.predict(transformed_new_data))
print(clf.predict_proba(transformed_new_data))

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import numpy as np

iris = datasets.load_iris()
data = iris.data[:, [0, 1]]
target = iris.target
target[100:125] = 1
target[125:150] = 0

xtrain = np.concatenate((data[0:40], data[50:90], data[100:140]))
ytrain = np.concatenate((target[0:40], target[50:90], target[100:140]))

xtest = np.concatenate((data[40:50], data[90:100], data[140:150]))
ytest = np.concatenate((target[40:50], target[90:100], target[140:150]))

clf = GaussianNB()
clf.fit(xtrain, ytrain)
pred = clf.predict(xtest)
predprob = clf.predict_proba(xtest)
print(pred)
print(predprob)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

print("Accuracy: ", accuracy_score(ytest, pred))
print("Precision: ", precision_score(ytest, pred, pos_label=1))
print("Recall: ", recall_score(ytest, pred, pos_label=1))
print("F1 Score: ", f1_score(ytest, pred, pos_label=1))

fpr, tpr, thresholds = roc_curve(ytest, predprob[:, 1])
print("AUC: ", auc(fpr, tpr))

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc(fpr, tpr))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
