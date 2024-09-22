import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import Isomap


# import data
srdata = pd.read_csv("./srdata.txt")

# question1 plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(srdata.V1, srdata.V2, srdata.V3)
ax.set_xlabel('V1')
ax.set_ylabel('V2')
ax.set_zlabel('V3')
plt.show()

# question2 apply ISOMAP
isomap = Isomap(n_neighbors = 4, n_components = 2)
isomap = isomap.fit(srdata)
transformed = pd.DataFrame(isomap.transform(srdata))
colors = [i - min(transformed.loc[:, 0].tolist()) + 1 for i in transformed.loc[:, 0].tolist()]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(srdata.V1, srdata.V2, srdata.V3, c=colors)
ax.set_xlabel('V1')
ax.set_ylabel('V2')
ax.set_zlabel('V3')
plt.show()
plt.scatter(transformed.loc[:, 0], transformed.loc[:, 1], c=colors)
plt.show()