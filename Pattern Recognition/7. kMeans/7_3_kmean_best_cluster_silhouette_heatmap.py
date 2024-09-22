from turtle import color, distance
import math
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

# import data
cdata = pd.read_csv("./cdata.txt")
target = cdata.loc[:, "Y"]
cdata = cdata.loc[:, ["X1", "X2"]]

# find best clusters number
# sse = []
# for i in range(1,11):
#     sse.append(KMeans(n_clusters=i, init=cdata.loc[0:i-1,:]).fit(cdata).inertia_)
# plt.plot(range(1,11), sse)
# plt.scatter(range(1,11), sse, marker='o')
# plt.show()

# apply k-mean
kmeans = KMeans(n_clusters=3, init=cdata.loc[0:2, :]).fit(cdata)
print("centers ", kmeans.cluster_centers_)
# print("labels ", kmeans.labels_)

# calculate cohesion and separation
print('cohesion ', kmeans.inertia_)
separation = 0
distance = lambda x1, x2: math.sqrt(((x1.X1 - x2.X1)**2) + ((x1.X2 - x2.X2)**2))
m = cdata.mean()
for i in list(set(kmeans.labels_)):
    mi = cdata.loc[kmeans.labels_==i, :].mean()
    Ci = len(cdata.loc[kmeans.labels_==i, :].index)
    separation += Ci * (distance(m, mi)**2)
print('separation', separation)

# plot data
# plt.scatter(cdata[(target==1)].X1, cdata[(target==1)].X2, c='red', marker='o')
# plt.scatter(cdata[(target==2)].X1, cdata[(target==2)].X2, c='blue', marker='o')
# plt.scatter(cdata[(target==3)].X1, cdata[(target==3)].X2, c='green', marker='o')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], marker='+', s=169, color="black")
# plt.show()

# calculate silhouette
print('total silhouette ', silhouette_score(cdata, kmeans.labels_))
print('silhouette cluster0 ', silhouette_samples(cdata, kmeans.labels_)[kmeans.labels_==0].mean())
print('silhouette cluster1 ', silhouette_samples(cdata, kmeans.labels_)[kmeans.labels_==1].mean())
print('silhouette cluster2 ', silhouette_samples(cdata, kmeans.labels_)[kmeans.labels_==2].mean())
# visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
# visualizer.fit(cdata)
# visualizer.show()

# create heatmap
cdata["cluster"] = kmeans.labels_
cdata = cdata.sort_values("cluster").drop("cluster", axis=1)
dist = distance_matrix(cdata, cdata)
plt.imshow(dist, cmap='hot')
plt.colorbar()
plt.show()