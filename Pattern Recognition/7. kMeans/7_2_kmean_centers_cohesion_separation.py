import math
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


# import data
X = [7,3,1,5,1,7,8,5]
Y = [1,4,5,8,3,8,2,9]
lables = ["x1","x2","x3","x4","x5","x6","x7","x8"]
kdata = pd.DataFrame({"X":X, "Y":Y}, index=lables)

# k-mean
kmeans = KMeans(n_clusters=3, init=kdata.loc[['x1','x2','x3'],:]).fit(kdata)
print("cluster centers: ", kmeans.cluster_centers_)
print("lables: ", kmeans.labels_)

print ('cohesion', kmeans.inertia_)

separation = 0
distance = lambda x1, x2: math.sqrt(((x1.X - x2.X)**2) + ((x1.Y - x2.Y)**2))
m = kdata.mean()
for i in list(set(kmeans.labels_)):
    mi = kdata.loc[kmeans.labels_ == i, :].mean()
    Ci = len(kdata.loc[kmeans.labels_ == i, :].index)
    separation += Ci * (distance(m,mi)**2)
print ("separatio", separation)

# plot data
plt.scatter(kdata.X, kdata.Y, c=kmeans.labels_)
for i in range(len(kdata.index)):
    plt.text(kdata.loc[lables[i], "X"], kdata.loc[lables[i], "Y"], '%s' %(str(lables[i])), size=15, zorder=1)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='+', s=169, c=range(3))
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
