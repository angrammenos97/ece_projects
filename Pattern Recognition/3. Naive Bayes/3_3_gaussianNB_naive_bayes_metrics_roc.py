import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# import data
iris = datasets.load_iris()
data = iris.data[:, [0, 1]]
target = iris.target
target[100:125] = 1
target[125:150] = 0

# split data
xtrain = np.concatenate((data[0:40], data[50:90], data[100:140]))
ytrain = np.concatenate((target[0:40], target[50:90], target[100:140]))
xtest = np.concatenate((data[40:50], data[90:100], data[140:150]))
ytest = np.concatenate((target[40:50], target[90:100], target[140:150]))

# question1 apply GaussianNB
clf = GaussianNB()
clf.fit(xtrain, ytrain)
pred = clf.predict(xtest)
predprob = clf.predict_proba(xtest)

# question2 metrics and ROC curve
print("Accuracy: ", accuracy_score(ytest, pred))
print("Precision: ", precision_score(ytest, pred, pos_label=1))
print("Recall: ", recall_score(ytest, pred, pos_label=1))
print("F1 Score: ", f1_score(ytest, pred, pos_label=1))
fpr, tpr, thresholds = roc_curve(ytest, predprob[:, 1])
print("AUC: ", auc(fpr, tpr))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc(fpr, tpr))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()