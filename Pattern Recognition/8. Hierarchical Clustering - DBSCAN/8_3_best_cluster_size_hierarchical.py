from matplotlib import projections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


# import data
europeData = pd.read_csv("./europe.txt")

# normalize data
scaler = StandardScaler()
scaler = scaler.fit(europeData)
europe = pd.DataFrame(scaler.transform(europeData), columns=europeData.columns, index=europeData.index)

# question1 hierachical clustering
clustering = AgglomerativeClustering(n_clusters=None, linkage='complete', distance_threshold=0).fit(europe)
linkage_matrix = np.column_stack([clustering.children_, clustering.distances_, np.ones(len(europe.index)-1)]).astype(float)
# dendrogram(linkage_matrix, labels=europe.index)
# plt.show()

# question2 find best cluster size
slc = []
for i in range(2,21):
    clustering = AgglomerativeClustering(n_clusters=i, linkage='complete').fit(europe)
    slc.append(silhouette_score(europe, clustering.labels_))
# plt.plot(range(2,21), slc)
# plt.xticks(range(2,21), range(2,21))
# plt.show()

#question3 clustering
clustering = AgglomerativeClustering(n_clusters=7, linkage='complete').fit(europe)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(europeData.GDP, europeData.Inflation, europeData.Unemployment, c=clustering.labels_, cmap='bwr')
# for i in range(len(europeData.index)):
#     ax.text(europeData.loc[europeData.index[i], "GDP"], europeData.loc[europeData.index[i], "Inflation"], europeData.loc[europeData.index[i], "Unemployment"], '%s' % (str(europeData.index[i])), size=20, zorder=1)
# ax.set_xlabel('GDP')
# ax.set_ylabel('Inflation')
# ax.set_zlabel('Unemployment')
# plt.show()

# question4 
print('Silhouette ', silhouette_score(europe, clustering.labels_))