import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, accuracy_score


data = pd.read_csv('./dcdata.txt')
target = data.loc[:,"Y"]
data = data.drop(["Y"], axis=1)

# single hierachical
clustering = AgglomerativeClustering(n_clusters=2, linkage='single').fit(data)
print('accuracy', accuracy_score(target, clustering.labels_))
# print('Silhouette ', silhouette_score(data, clustering.labels_))

# complete hierachical
clustering = AgglomerativeClustering(n_clusters=2, linkage='complete').fit(data)
print('accuracy', accuracy_score(target, clustering.labels_))
# print('Silhouette ', silhouette_score(data, clustering.labels_))

#DBSCAN 0.75
clustering = DBSCAN(eps=0.75, min_samples=5).fit(data)
print('accuracy', accuracy_score(target, clustering.labels_))

#DBSCAN 1
clustering = DBSCAN(eps=1, min_samples=5).fit(data)
print('accuracy', accuracy_score(target, clustering.labels_))

#DBSCAN 1.25
clustering = DBSCAN(eps=1.25, min_samples=5).fit(data)
print('accuracy', accuracy_score(target, clustering.labels_))

#DBSCAN 1.5
clustering = DBSCAN(eps=1.5, min_samples=5).fit(data)
print('accuracy', accuracy_score(target, clustering.labels_))

# k-Means
kmeans = KMeans(n_clusters=2).fit(data)
print('accuracy', accuracy_score(target, kmeans.labels_))
# plt.scatter(data.X1, data.X2, c=kmeans.labels_)
# plt.show()
