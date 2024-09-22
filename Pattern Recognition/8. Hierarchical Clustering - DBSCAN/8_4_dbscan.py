import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster

from sklearn.cluster import DBSCAN


#question1 import data and plot
X = [2, 2, 8, 5, 7, 6, 1, 4]
Y = [10, 5, 4, 8, 5, 4, 2, 9]
labels = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
ddata = pd.DataFrame({"X": X, "Y": Y}, index=labels)
# plt.scatter(ddata.X, ddata.Y)
# for i in range(len(ddata.index)):
#     plt.text(ddata.loc[labels[i], "X"], ddata.loc[labels[i], "Y"], '%s' % (str(labels[i])), size=15, zorder=1)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

#question3 DBSCAN
clustering = DBSCAN(eps=2, min_samples=2).fit(ddata)
clusters = clustering.labels_
plt.scatter(ddata.X, ddata.Y, c=clusters, cmap="spring")
for i in range(len(ddata.index)):
    plt.text(ddata.loc[labels[i], "X"], ddata.loc[labels[i], "Y"], '%s' % (str(labels[i])), size=15, zorder=1)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("DBSCAN(eps=2, minPts=2)")
plt.show()
