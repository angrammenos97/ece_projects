import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor


# question1 import data
anndata = pd.DataFrame({"X1": [0, 0, 1, 1], "X2": [0, 1, 0, 1], "Y": [1, 1, -1, -1]})
X = anndata.loc[:, ["X1", "X2"]]
y = anndata.loc[:, "Y"]

# question2 plot and apply MLP regressor
plt.scatter(anndata[(anndata.Y == -1)].X1, anndata[(anndata.Y == -1)].X2, c="red", marker="+")
plt.scatter(anndata[(anndata.Y == 1)].X1, anndata[(anndata.Y == 1)].X2, c="blue", marker="o")
plt.show()
clf = MLPRegressor(hidden_layer_sizes=(), learning_rate_init=1)
clf = clf.fit(X, y)
print(clf.predict([[0, 0]]))
print(clf.predict([[1, 0]]))
print(clf.predict([[0.5, 0]]))