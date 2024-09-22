from numpy import size
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


# import data
X = [2, 8, 0, 7, 6]
Y = [0, 4, 6, 2, 1]
labels = ["x1", "x2", "x3", "x4", "x5"]
hdata = pd.DataFrame({"X": X, "Y": Y}, index=labels)

# plot
# plt.scatter(hdata.X, hdata.Y)
# for i in range(len(hdata.index)):
#     plt.text(hdata.loc[labels[i], "X"], hdata.loc[labels[i], "Y"], '%s' %(str(labels[i])), size=15, zorder=1)
# plt.show()

# Single Linkage Hierarchical Clustering
clustering = AgglomerativeClustering(n_clusters=None, linkage='single', distance_threshold=0).fit(hdata)
linkage_matrix = np.column_stack([clustering.children_, clustering.distances_, np.ones(len(hdata.index)-1)]).astype(float)
# dendrogram(linkage_matrix, labels=labels)
#plt.show()

# Complete Linkage Hierarchical Clustering
clustering = AgglomerativeClustering(n_clusters=None, linkage='complete', distance_threshold=0).fit(hdata)
linkage_matrix = np.column_stack([clustering.children_, clustering.distances_, np.ones(len(hdata.index)-1)]).astype(float)
# dendrogram(linkage_matrix, labels=labels)
#plt.show()

# Single Linkage Hierarchical Clustering (2 clusters)
clustering = AgglomerativeClustering(n_clusters=2, linkage='single').fit(hdata)
# plt.scatter(hdata.X, hdata.Y, c=clustering.labels_, cmap='bwr')
# for i in range(len(hdata.index)):
#     plt.text(hdata.loc[labels[i], "X"], hdata.loc[labels[i], "Y"], '%s' %(str(labels[i])), size=15, zorder=1)
# plt.show()

# Complete Linkage Hierarchical Clustering (2 clusters)
clustering = AgglomerativeClustering(n_clusters=2, linkage='complete').fit(hdata)
plt.scatter(hdata.X, hdata.Y, c=clustering.labels_, cmap='bwr')
for i in range(len(hdata.index)):
    plt.text(hdata.loc[labels[i], "X"], hdata.loc[labels[i], "Y"], '%s' %(str(labels[i])), size=15, zorder=1)
plt.show()