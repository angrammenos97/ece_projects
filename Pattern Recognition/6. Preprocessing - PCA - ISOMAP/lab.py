import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt

# X = [1, 0, -1, 0, -1, 1]
# Y = [0, 1, 1, -1, 0, -1]
# Z = [-1, -1, 0, 1, 1, 0]

# labels = ["x1", "x2", "x3", "x4", "x5", "x6"]

# pdata = pd.DataFrame({"X":X, "Y":Y, "Z":Z}, index=labels)

# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# ax.scatter(pdata.X, pdata.Y, pdata.Z)

# for i in range(len(pdata.index)):
#     ax.text(pdata.loc[labels[i], "X"], pdata.loc[labels[i], "Y"], pdata.loc[labels[i], "Z"], '%s' % (str(labels[i])), size=20, zorder=1)

# ax.set_xlabel('X_Axis')
# ax.set_ylabel('Y_Axis')
# ax.set_zlabel('Z_Axis')

# plt.show()

# engdata = pd.read_csv("./engdata.txt")

# pdata = engdata.loc[:, ["Age", "Salary"]]
# pdata = pdata.drop_duplicates()

# scaler = StandardScaler()
# scaler.fit(pdata)
# transformed = pd.DataFrame(scaler.transform(pdata), columns=["Age", "Salary"])

# plt.scatter(pdata["Age"], pdata['Salary'])
# plt.scatter(transformed["Age"], transformed["Salary"])

# data_sample = pdata.sample(n=150, random_state=1, replace=True)
# plt.scatter(data_sample["Age"], data_sample['Salary'])
# plt.show()
 
# engdata = engdata.loc[:, ["Age","Salary","YearsOfStudy","WorkExp"]]

# scaler = StandardScaler()
# scaler.fit(engdata)

# transformed = pd.DataFrame(scaler.transform(engdata), columns=engdata.columns)

# pca = PCA()
# pca.fit(transformed)

# pca_transformed = pca.transform(transformed)

# eigenvalues = pca.explained_variance_
# eigenvectors = pca.components_

# print(eigenvalues)
# print(eigenvectors)

# plt.bar(range(len(eigenvalues)), eigenvalues/sum(eigenvalues))
# plt.show()

srdata = pd.read_csv("./srdata.txt")

isomap = Isomap(n_neighbors=4, n_components=2)
isomap = isomap.fit(srdata)
transformed = pd.DataFrame(isomap.transform(srdata))

colors = [i-min(transformed.loc[:,0].tolist())+1 for i in transformed.loc[:,0].tolist()]

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(srdata.V1, srdata.V2, srdata.V3, c=colors)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

plt.scatter(transformed.loc[:,0], transformed.loc[:,1], c=colors)
plt.show()
