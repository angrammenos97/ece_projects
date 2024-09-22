import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


kmdata = pd.read_csv("./kmdata.txt")

plt.scatter(kmdata.X1, kmdata.X2)
plt.show()

plt.scatter(kmdata.X1, kmdata.X2, c=kmdata.Y)
plt.show()

target = kmdata.loc[:, ["Y"]]
data = kmdata.drop("Y", axis=1)

kmeans = KMeans(n_clusters=3).fit(data)
plt.scatter(kmdata.X1, kmdata.X2, c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='black', marker='+')
plt.show()

gm = GaussianMixture(n_components=3, tol=0.0001).fit(data)
plt.scatter(kmdata.X1, kmdata.X2, c=gm.predict(data))
plt.scatter(gm.means_[:,0], gm.means_[:,1], c='black', marker='+')
plt.show()