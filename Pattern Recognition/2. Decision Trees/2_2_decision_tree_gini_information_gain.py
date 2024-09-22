import math
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.preprocessing import OneHotEncoder


# question1 import data
weather = pd.read_csv("./weather.txt")

# question2 decision criterias
encoder = OneHotEncoder()
##from GINI
#Outlook
encoder.fit(weather.loc[:, ['Outlook']])
transformedOutlook = encoder.transform(weather.loc[:, ['Outlook']])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(transformedOutlook, weather.loc[:, 'Play'])
fig = plt.figure()
tree.plot_tree(clf, class_names=['No', 'Yes'], filled=True)
plt.show()
#Temperature
encoder.fit(weather.loc[:, ['Temperature']])
transformedTemperature = encoder.transform(weather.loc[:, ['Temperature']])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(transformedTemperature, weather.loc[:, 'Play'])
fig = plt.figure()
tree.plot_tree(clf, class_names=['No', 'Yes'], filled=True)
plt.show()
#Humidity
encoder.fit(weather.loc[:, ['Humidity']])
transformedHumidity = encoder.transform(weather.loc[:, ['Humidity']])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(transformedHumidity, weather.loc[:, 'Play'])
fig = plt.figure()
tree.plot_tree(clf, class_names=['No', 'Yes'], filled=True)
plt.show()
##from Information Gain
freq = pd.crosstab("Play", weather.Play, normalize="index")
EntropyAll = - freq.No * math.log2(freq.No) - freq.Yes * math.log2(freq.Yes)
#Outlook
absfreq = pd.crosstab(weather.Outlook, weather.Play)
freq = pd.crosstab(weather.Outlook, weather.Play, normalize='index')
freqSum = pd.crosstab(weather.Outlook, weather.Play, normalize='all').sum(axis=1)
EntropySunny = - freq.loc['Sunny', 'No'] * math.log2(freq.loc['Sunny', 'No']) - freq.loc['Sunny', 'Yes'] * math.log2(freq.loc['Sunny', 'Yes'])
EntropyRainy = - freq.loc['Rainy', 'No'] * math.log2(freq.loc['Rainy', 'No']) - freq.loc['Rainy', 'Yes'] * math.log2(freq.loc['Rainy', 'Yes'])
GAINOutlook = EntropyAll - freqSum.loc['Sunny'] * EntropySunny - freqSum.loc['Rainy'] * EntropyRainy
print("Gain Outlook = ", GAINOutlook)
#Temperature
absfreq = pd.crosstab(weather.Temperature, weather.Play)
freq = pd.crosstab(weather.Temperature, weather.Play, normalize='index')
freqSum = pd.crosstab(weather.Temperature, weather.Play, normalize='all').sum(axis=1)
EntropyHot = - freq.loc['Hot', 'No'] * math.log2(freq.loc['Hot', 'No']) - freq.loc['Hot', 'Yes'] * math.log2(freq.loc['Hot', 'Yes'])
EntropyCool = - freq.loc['Cool', 'No'] * math.log2(freq.loc['Cool', 'No']) - freq.loc['Cool', 'Yes'] * math.log2(freq.loc['Cool', 'Yes'])
GAINTemperature = EntropyAll - freqSum.loc['Hot'] * EntropyHot - freqSum.loc['Cool'] * EntropyCool
print("Gain Temperature = ", GAINTemperature)
#Humidity
absfreq = pd.crosstab(weather.Humidity, weather.Play)
freq = pd.crosstab(weather.Humidity, weather.Play, normalize='index')
freqSum = pd.crosstab(weather.Humidity, weather.Play, normalize='all').sum(axis=1)
EntropyHigh = - freq.loc['High', 'No'] * math.log2(freq.loc['High', 'No']) - freq.loc['High', 'Yes'] * math.log2(freq.loc['High', 'Yes'])
EntropyLow = - freq.loc['Low', 'No'] * math.log2(freq.loc['Low', 'No']) - freq.loc['Low', 'Yes'] * math.log2(freq.loc['Low', 'Yes'])
GAINHumidity = EntropyAll - freqSum.loc['High'] * EntropyHigh - freqSum.loc['Low'] * EntropyLow
print("Gain Humidity = ", GAINHumidity)

# question3 create decision tree and plot
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
encoder.fit(weather.loc[:, ['Outlook', 'Temperature', 'Humidity']])
transformed = encoder.transform(weather.loc[:, ['Outlook', 'Temperature', 'Humidity']])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(transformed, weather.loc[:, 'Play'])
fig = plt.figure(figsize=(10, 9))
tree.plot_tree(clf, class_names=['No', 'Yes'], filled=True)
plt.show()
text_representation = tree.export_text(clf)
print(text_representation)