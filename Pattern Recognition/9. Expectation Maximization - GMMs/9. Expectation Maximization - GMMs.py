import pandas as pd
gdata = pd.read_csv("./gdata.txt")
x = gdata.loc[:, "X"]
y = gdata.loc[:, "Y"]

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.scatter(x[(y == 1)], np.zeros(len(x[(y == 1)].tolist())), c="red", marker="+")
plt.scatter(x[(y == 2)], np.zeros(len(x[(y == 2)].tolist())), c="blue", marker="o")
sns.histplot(x, kde=True, stat="density", linewidth=0, fill=False)
plt.show()

from scipy.stats import norm
import math
mu = [0, 1]
lamda = [0.5, 0.5]
epsilon = 1e-08
log_likelihood = np.sum([math.log(i) for i in lamda[0] * norm.pdf(x, loc=mu[0], scale=1) + lamda[1] * norm.pdf(x, loc=mu[1], scale=1)])

while True:
    T1 = norm.pdf(x, loc=mu[0], scale=1)
    T2 = norm.pdf(x, loc=mu[1], scale=1)
    P1 = lamda[0] * T1 / (lamda[0] * T1 + lamda[1] * T2)
    P2 = lamda[1] * T2 / (lamda[0] * T1 + lamda[1] * T2)

    mu[0] = np.sum(P1 * x) / np.sum(P1)
    mu[1] = np.sum(P2 * x) / np.sum(P2)
    lamda[0] = np.mean(P1)
    lamda[1] = np.mean(P2)

    new_log_likelihood = np.sum([math.log(i) for i in lamda[0] * norm.pdf(x, loc=mu[0], scale=1) + lamda[1] * norm.pdf(x, loc=mu[1], scale=1)])

    print("mu=", mu, " lambda=", lamda, " log_likelihood=", new_log_likelihood)
    if ((new_log_likelihood - log_likelihood) <= epsilon): break
    log_likelihood = new_log_likelihood


from sklearn.mixture import GaussianMixture
data = np.array(x.tolist()).reshape(-1,1)
gm = GaussianMixture(n_components=2).fit(data)
print(gm.means_)
print(gm.covariances_)
print(np.sum(gm.score_samples(data)))

def mix_pdf(x, loc, scale, weights):
    d = np.zeros_like(x)
    for mu, sigma, pi in zip(loc, scale, weights):
        d += pi * norm.pdf(x, loc = mu, scale = sigma)
    return d

pi, mu, sigma = gm.weights_.flatten(), gm.means_.flatten(), np.sqrt(gm.covariances_.flatten())
grid = np.arange(np.min(x), np.max(x), 0.01)
plt.scatter(x[(y == 1)], np.zeros(len(x[(y == 1)].tolist())), c="red", marker="+")
plt.scatter(x[(y == 2)], np.zeros(len(x[(y == 2)].tolist())), c="blue", marker="o")
plt.plot(grid, mix_pdf(grid, mu, sigma, pi), label="GMM")
sns.histplot(x, kde=True, stat="density", linewidth=0, fill=False)
plt.legend(loc = 'upper right')
plt.show()


gsdata = pd.read_csv("./gsdata.txt")
target = gsdata.loc[:, "Y"]
gsdata = gsdata.drop(["Y"], axis=1)

plt.scatter(gsdata[(target == 1)].X1, gsdata[(target == 1)].X2, c="red", marker="+")
plt.scatter(gsdata[(target == 2)].X1, gsdata[(target == 2)].X2, c="green", marker="o")
plt.scatter(gsdata[(target == 3)].X1, gsdata[(target == 3)].X2, c="blue", marker="x")
plt.show()

gm = GaussianMixture(n_components=3, tol=0.1).fit(gsdata)

x = np.linspace(np.min(gsdata.loc[:, "X1"]), np.max(gsdata.loc[:, "X1"]))
y = np.linspace(np.min(gsdata.loc[:, "X2"]), np.max(gsdata.loc[:, "X2"]))
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gm.score_samples(XX)
Z = Z.reshape(X.shape)

plt.contour(X, Y, Z)
plt.scatter(gsdata[(target == 1)].X1, gsdata[(target == 1)].X2, c="red", marker="+")
plt.scatter(gsdata[(target == 2)].X1, gsdata[(target == 2)].X2, c="green", marker="o")
plt.scatter(gsdata[(target == 3)].X1, gsdata[(target == 3)].X2, c="blue", marker="x")
plt.show()

clusters = gm.predict(gsdata)
centers = gm.means_

from sklearn.metrics import silhouette_score
print(silhouette_score(gsdata, clusters))

gsdata["cluster"] = clusters
gsdata = gsdata.sort_values("cluster").drop("cluster", axis=1)
from scipy.spatial import distance_matrix
dist = distance_matrix(gsdata, gsdata)
plt.imshow(dist, cmap='hot')
plt.colorbar()
plt.show()


icdata = pd.read_csv("./icdata.txt")
x = icdata.loc[:, "X"]
y = icdata.loc[:, "Y"]
data = np.array(x.tolist()).reshape(-1,1)

plt.scatter(x[(y == 1)], np.zeros(len(x[(y == 1)].tolist())), c="red", marker="+")
plt.scatter(x[(y == 2)], np.zeros(len(x[(y == 2)].tolist())), c="green", marker="o")
plt.scatter(x[(y == 3)], np.zeros(len(x[(y == 3)].tolist())), c="blue", marker="x")
sns.histplot(x, kde=True, stat="density", linewidth=0, fill=False)
plt.show()

fig, axs = plt.subplots(4, 1)
fig.tight_layout()
n = [2, 3, 4, 5]
AIC = []
BIC = []
for i in n:
    gm = GaussianMixture(n_components=i).fit(data)
    pi, mu, sigma = gm.weights_.flatten(), gm.means_.flatten(), np.sqrt(gm.covariances_.flatten())
    grid = np.arange(np.min(x), np.max(x), 0.01)
    axs[i - 2].scatter(x[(y == 1)], np.zeros(len(x[(y == 1)].tolist())), c="red", marker="+")
    axs[i - 2].scatter(x[(y == 2)], np.zeros(len(x[(y == 2)].tolist())), c="green", marker="o")
    axs[i - 2].scatter(x[(y == 3)], np.zeros(len(x[(y == 3)].tolist())), c="blue", marker="x")
    axs[i - 2].plot(grid, mix_pdf(grid, mu, sigma, pi), label="GMM")
    axs[i - 2].set_title("Density Curves (k=" + str(i) + ")")
    axs[i - 2].set_xlabel("Data")
    axs[i - 2].set_ylabel("Density")
    AIC.append(gm.aic(data))
    BIC.append(gm.bic(data))
plt.show()

plt.plot(n, AIC)
plt.title("AIC")
plt.xlabel("Index")
plt.ylabel("AIC")
plt.show()

plt.plot(n, BIC)
plt.title("BIC")
plt.xlabel("Index")
plt.ylabel("BIC")
plt.show()