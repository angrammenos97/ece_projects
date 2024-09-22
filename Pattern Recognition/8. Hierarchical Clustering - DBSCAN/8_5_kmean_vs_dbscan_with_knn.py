from turtle import distance
from numpy import indices
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors


# import data
mdata = pd.read_csv("./mdata.txt")

# question1 k-means
kmeans = KMeans(n_clusters=2).fit(mdata)
# plt.scatter(mdata.X, mdata.Y, c=kmeans.labels_, cmap='bwr')
# plt.show()

#question2 find eps from kNN-distance
nbrs = NearestNeighbors(n_neighbors=10).fit(mdata)
distances, indices = nbrs.kneighbors(mdata)
distanceDec = sorted(distances[:,9])
# plt.plot(distanceDec)
# plt.ylabel("10-NN Distance")
# plt.xlabel("Points sorted by distance")
# plt.show()

#question3 DBSCAN
clustering = DBSCAN(eps=0.4, min_samples=10).fit(mdata)
plt.scatter(mdata.X, mdata.Y, c=clustering.labels_)
plt.show()