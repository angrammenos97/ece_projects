import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import CategoricalNB


# question1 import data
traffic = pd.read_csv("./traffic.txt")

# question3 build Naive Bayes
X = traffic.loc[:, ["Weather", "Day"]]
y = traffic.loc[:, "HighTraffic"]
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
encoder = encoder.fit(X)
X = encoder.transform(X)
#alpha=0
clf = CategoricalNB(alpha = 0)
clf.fit(X, y)
new_data = pd.DataFrame({"Weather": ["Hot"], "Day": ["Vacation"]})
transformed_new_data = encoder.transform(new_data)
print(clf.predict(transformed_new_data))
print(clf.predict_proba(transformed_new_data))
#alpha=1
clf = CategoricalNB(alpha=1)
clf.fit(X, y)
new_data = pd.DataFrame({"Weather": ["Hot"], "Day": ["Weekend"]})
transformed_new_data = encoder.transform(new_data)
print(clf.predict(transformed_new_data))
print(clf.predict_proba(transformed_new_data))
