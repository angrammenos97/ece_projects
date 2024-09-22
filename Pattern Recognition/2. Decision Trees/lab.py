from numpy.core.arrayprint import SubArrayFormat
import pandas as pd
from scipy.sparse import data
from sklearn import tree
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import math

weather = pd.read_csv("./weather.txt")

# absfreq = pd.crosstab(weather.Outlook, weather.Play)
# freq = pd.crosstab(weather.Outlook, weather.Play, normalize='index')
# freqSumOutLook = pd.crosstab(weather.Outlook, weather.Play, normalize='all').sum(axis=1)
# freqSumPlay = pd.crosstab(weather.Outlook, weather.Play, normalize='all').sum(axis=0)

# GINI_Sunny = 1 - freq.loc['Sunny', 'No']**2 - freq.loc['Sunny', 'Yes']**2
# GINI_Rainy = 1 - freq.loc['Rainy', 'No']**2 - freq.loc['Rainy', 'Yes']**2
# GINI_Outlook = freqSumOutLook.loc['Sunny']*GINI_Sunny + freqSumOutLook.loc['Rainy']*GINI_Rainy
# # print(GINI_Outlook)

# EntropyAll = - freqSumPlay.loc['No']*math.log2(freqSumPlay.loc['No']) - freqSumPlay.loc['Yes']*math.log2(freqSumPlay.loc['Yes'])
# EntropySunny = - freq.loc['Sunny','No']*math.log2(freq.loc['Sunny','No']) - freq.loc['Sunny','Yes']*math.log2(freq.loc['Sunny','Yes'])
# EntropyRainy = - freq.loc['Rainy','No']*math.log2(freq.loc['Rainy','No']) - freq.loc['Rainy','Yes']*math.log2(freq.loc['Rainy','Yes'])
# GainOutlook = EntropyAll - freqSumOutLook.loc['Sunny']*EntropySunny - freqSumOutLook.loc['Rainy']*EntropyRainy
# # print(GainOutlook)

# encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
# encoder.fit(weather.loc[:, ['Outlook', 'Temperature', 'Humidity']])

# tfOutlook = encoder.transform(weather.loc[:,['Outlook',  'Temperature', 'Humidity']])
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(tfOutlook, weather.loc[:, 'Play'])

# fig = plt.figure()
# tree.plot_tree(clf, class_names=['No', 'Yes'], filled=True)
# plt.show()

# new_data = pd.DataFrame({"Outlook": ["Sunny"], "Temperature": ["Cold"], "Humidity": ["High"]})
# tf_new_data = encoder.transform(new_data)
# print(clf.predict(tf_new_data))
# print(clf.predict_proba(tf_new_data))

iris = datasets.load_iris()
data = iris.data[:, [0,1]]
target = iris.target
target[100:125] = 0
target[125:150] = 1

xtrain = np.concatenate((data[0:40], data[50:90], data[100:140]))
xtest = np.concatenate((data[40:50], data[90:100], data[140:150]))
ytrain = np.concatenate((target[0:40], target[50:90], target[100:140]))
ytest = np.concatenate((target[40:50], target[90:100], target[140:150]))

clf = tree.DecisionTreeClassifier(min_samples_split=20)
clf = clf.fit(xtrain, ytrain)
pred = clf.predict(xtest)

print(confusion_matrix(ytest, pred))
print(accuracy_score(ytest, pred))
print(precision_score(ytest, pred))
print(recall_score(ytest, pred))
print(f1_score(ytest, pred))