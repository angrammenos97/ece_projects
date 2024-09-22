import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# import data
engdata = pd.read_csv("./engdata.txt")
location = engdata.Location
engdata = engdata.drop(["Location"], axis=1)

# question1 plot and find correlation
plt.scatter(engdata[(location == "EU")].Age, engdata[(location == "EU")].Salary, c="red", marker="+")
plt.scatter(engdata[(location == "US")].Age, engdata[(location == "US")].Salary, c="blue", marker="o")
plt.show()
print(engdata.corr())

# question2 apply PCA
scaler = StandardScaler()
scaler = scaler.fit(engdata)
transformed = pd.DataFrame(scaler.transform(engdata), columns=engdata.columns)
pca = PCA()
pca = pca.fit(transformed)
pca_transformed = pca.transform(transformed)
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_
plt.bar(range(len(eigenvalues)), eigenvalues/sum(eigenvalues))
plt.show()
pca = PCA(n_components=2)
pca_transformed = pd.DataFrame(pca.fit_transform(transformed))
plt.scatter(pca_transformed.loc[:, 0], pca_transformed.loc[:, 1])
plt.show()
pca_inverse = pd.DataFrame(pca.inverse_transform(pca_transformed), columns=engdata.columns)
plt.scatter(pca_inverse[(location == "EU")].Age, pca_inverse[(location == "EU")].Salary, c="red", marker="+")
plt.scatter(pca_inverse[(location == "US")].Age, pca_inverse[(location == "US")].Salary, c="blue", marker="o")
plt.show()
info_gain = (eigenvalues[2] + eigenvalues[3]) / sum(eigenvalues)
print(info_gain)