import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


# import data
gsdata = pd.read_csv("./gsdata.txt")
target = gsdata.loc[:, "Y"]
gsdata = gsdata.drop(["Y"], axis=1)

# question1 plot
# plt.scatter(gsdata[(target == 1)].X1, gsdata[(target == 1)].X2, c="red", marker="+")
# plt.scatter(gsdata[(target == 2)].X1, gsdata[(target == 2)].X2, c="green", marker="o")
# plt.scatter(gsdata[(target == 3)].X1, gsdata[(target == 3)].X2, c="blue", marker="x")
# plt.show()

# question2 create Gaussian Mixture model
gm = GaussianMixture(n_components=3, tol=0.1).fit(gsdata)
# x = np.linspace(np.min(gsdata.loc[:, "X1"]), np.max(gsdata.loc[:, "X1"]))
# y = np.linspace(np.min(gsdata.loc[:, "X2"]), np.max(gsdata.loc[:, "X2"]))
# X, Y = np.meshgrid(x, y)
# XX = np.array([X.ravel(), Y.ravel()]).T
# Z = -gm.score_samples(XX)
# Z = Z.reshape(X.shape)
# plt.contour(X, Y, Z)
# plt.scatter(gsdata[(target == 1)].X1, gsdata[(target == 1)].X2, c="red", marker="+")
# plt.scatter(gsdata[(target == 2)].X1, gsdata[(target == 2)].X2, c="green", marker="o")
# plt.scatter(gsdata[(target == 3)].X1, gsdata[(target == 3)].X2, c="blue", marker="x")
# plt.show()
clusters = gm.predict(gsdata)
centers = gm.means_
print("Centers = ", centers)

# question3 calculate Silhouette
print("Silhoutte = ", silhouette_score(gsdata, clusters))

# question4 create Heatmap
gsdata["cluster"] = clusters
gsdata = gsdata.sort_values("cluster").drop("cluster", axis=1)
dist = distance_matrix(gsdata, gsdata)
plt.imshow(dist, cmap="hot")
plt.colorbar()
plt.show()