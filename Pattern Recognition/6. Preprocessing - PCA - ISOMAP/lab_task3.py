import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# question1 import data
X = [1, 0, -1, 0, -1, 1]
Y = [0, 1, 1, -1, 0, -1]
Z = [-1, -1, 0, 1, 1, 0]
labels = ["x1", "x2", "x3", "x4", "x5", "x6"]
pdata = pd.DataFrame({"X": X, "Y": Y, "Z": Z}, index=labels)

# question2 apply PCA
# pca = PCA(n_components=None)
# pca = pca.fit(pdata)
# transformed = pca.transform(pdata)
# eigenvalues = pca.explained_variance_
# eigenvectors = pca.components_
# print("Eigen values = ", eigenvalues)
# print("Eigen vectors = ", eigenvectors)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(pdata.X, pdata.Y, pdata.Z)
for i in range(len(pdata.index)):
    ax.text(pdata.loc[labels[i], "X"], pdata.loc[labels[i], "Y"],
pdata.loc[labels[i], "Z"], '%s' % (str(labels[i])), size=20, zorder=1)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.show()