import math
import pandas as pd


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


data = pd.read_csv("./quiz_data (1).csv")

# question1
centers = pd.DataFrame({"X1":[-4, 0, 4], "X2":[10, 0, 10]})
kmeans = KMeans(n_clusters=3, init=centers).fit(data)
print('cohesion ', kmeans.inertia_)

# question2
separation = 0
distance = lambda x1,x2: math.sqrt(((x1.X1 - x2.X1)**2)+((x1.X2 - x2.X2)**2))
m = data.mean()
for i in list(set(kmeans.labels_)):
    mi = data.loc[kmeans.labels_==i,:].mean()
    Ci = len(data.loc[kmeans.labels_==i,:].index)
    separation += Ci * (distance(m, mi)**2)
print('separation ', separation)

#question3
print('silhouette ', silhouette_score(data, kmeans.labels_))

# question4
centers = pd.DataFrame({"X1":[-2, 2, 0], "X2":[0, 0, 10]})
kmeans = KMeans(n_clusters=3, init=centers).fit(data)
print('cohesion ', kmeans.inertia_)

separation = 0
distance = lambda x1,x2: math.sqrt(((x1.X1 - x2.X1)**2)+((x1.X2 - x2.X2)**2))
m = data.mean()
for i in list(set(kmeans.labels_)):
    mi = data.loc[kmeans.labels_==i,:].mean()
    Ci = len(data.loc[kmeans.labels_==i,:].index)
    separation += Ci * (distance(m, mi)**2)
print('separation ', separation)

print('silhouette ', silhouette_score(data, kmeans.labels_))