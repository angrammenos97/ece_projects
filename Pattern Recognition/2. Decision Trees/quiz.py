import pandas as pd
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

data = pd.read_csv("quiz_data.csv")

encoder = OneHotEncoder()
encoder.fit(data.loc[:,['CustomerID','Sex','CarType','Budget']])
tf_data = encoder.transform(data.loc[:,['CustomerID','Sex','CarType','Budget']])
clf = tree.DecisionTreeClassifier()
clf.fit(tf_data, data.loc[:, 'Insurance'])
fig = plt.figure
tree.plot_tree(clf, class_names=['Yes','No'])
plt.show()
##############

##############
absfreq = pd.crosstab(data.Sex, data.Insurance)
freq = pd.crosstab(data.Sex, data.Insurance, normalize='index')
freqSum = pd.crosstab(data.Sex, data.Insurance, normalize='all').sum(axis=1)
GINI_Male = 1 - freq.loc["M", "No"]**2 - freq.loc["M", "Yes"]**2
print(GINI_Male)
##############
absfreq = pd.crosstab(data.CarType, data.Insurance)
freq = pd.crosstab(data.CarType, data.Insurance, normalize='index')
freqSum = pd.crosstab(data.CarType, data.Insurance, normalize='all').sum(axis=1)
GINI_Family = 1 - freq.loc["Family", "No"]**2 - freq.loc["Family", "Yes"]**2
GINI_Sport = 1 - freq.loc["Sport", "No"]**2 - freq.loc["Sport", "Yes"]**2
GINI_Sedan = 1 - freq.loc["Sedan", "No"]**2 - freq.loc["Sedan", "Yes"]**2
GINI_CarType = freqSum.loc["Family"] * GINI_Family + freqSum["Sport"] * GINI_Sport + freqSum["Sedan"] * GINI_Sedan
print(GINI_CarType)
##############
absfreq = pd.crosstab(data.Budget, data.Insurance)
freq = pd.crosstab(data.Budget, data.Insurance, normalize='index')
freqSum = pd.crosstab(data.Budget, data.Insurance, normalize='all').sum(axis=1)
GINI_Low = 1 - freq.loc["Low", "No"]**2 - freq.loc["Low", "Yes"]**2
GINI_Medium = 1 - freq.loc["Medium", "No"]**2 - freq.loc["Medium", "Yes"]**2
GINI_High = 1 - freq.loc["High", "No"]**2 - freq.loc["High", "Yes"]**2
GINI_VeryHigh = 1 - freq.loc["VeryHigh", "No"]**2 - freq.loc["VeryHigh", "Yes"]**2
GINI_Budget = freqSum.loc["Low"] * GINI_Low + freqSum.loc["Medium"] * GINI_Medium + freqSum["High"] * GINI_High + freqSum["VeryHigh"] * GINI_VeryHigh
print(GINI_Budget)

# encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
# encoder.fit(data.loc[:, ['CustomerID', 'Sex', 'CarType','Budget']])

# tf_data = encoder.transform(data.loc[:,['CustomerID', 'Sex', 'CarType','Budget']])
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(tf_data, data.loc[:, 'Insurance'])

# # fig = plt.figure()
# # tree.plot_tree(clf, class_names=['No', 'Yes'], filled=True)
# # plt.show()

# absfreq = pd.crosstab(data.Budget, data.Insurance)
# freq = pd.crosstab(data.Budget, data.Insurance, normalize='index')
# freqSumBudget = pd.crosstab(data.Budget, data.Insurance, normalize='all').sum(axis=1)
# freqSumPlay = pd.crosstab(data.Budget, data.Insurance, normalize='all').sum(axis=0)

# GINI_Low = 1 - freq.loc['Low', 'No']**2 - freq.loc['Low', 'Yes']**2
# GINI_Medium = 1 - freq.loc['Medium', 'No']**2 - freq.loc['Medium', 'Yes']**2
# GINI_High = 1 - freq.loc['High', 'No']**2 - freq.loc['High', 'Yes']**2
# GINI_VeryHigh = 1 - freq.loc['VeryHigh', 'No']**2 - freq.loc['VeryHigh', 'Yes']**2
# GINI_Budget = freqSumBudget.loc['Low']*GINI_Low + freqSumBudget.loc['Medium']*GINI_Medium + freqSumBudget.loc['High']*GINI_High + freqSumBudget.loc['VeryHigh']*GINI_VeryHigh
# print(GINI_Budget)