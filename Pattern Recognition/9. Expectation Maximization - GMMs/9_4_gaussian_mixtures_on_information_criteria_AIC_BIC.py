import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture


# import data
icdata = pd.read_csv("./icdata.txt")
x = icdata.loc[:, "X"]
y = icdata.loc[:, "Y"]

# question1 plot
# plt.scatter(x[(y==1)], np.zeros(len(x[(y==1)].tolist())), c='red', marker='+')
# plt.scatter(x[(y==2)], np.zeros(len(x[(y==2)].tolist())), c='green', marker='o')
# plt.scatter(x[(y==3)], np.zeros(len(x[(y==3)].tolist())), c='blue', marker='x')
# sns.histplot(x, kde=True, stat="density", linewidth=0, fill=False)
# plt.show()

# question2 number of distribution according Information Criteria
fig, axs = plt.subplots(4,1)
fig.tight_layout()
n = [2, 3, 4, 5]
AIC = []
BIC = []
data = np.array(x.tolist()).reshape(-1,1)
def mix_pdf(x, loc, scale, weights):
    d = np.zeros_like(x)
    for mu, sigma, pi in zip(loc, scale, weights):
        d += pi * norm.pdf(x, loc=mu, scale=sigma)
    return d
for i in n:
    gm = GaussianMixture(n_components=i).fit(data)
    pi, mu, sigma = gm.weights_.flatten(), gm.means_.flatten(), np.sqrt(gm.covariances_.flatten())
    grid = np.arange(np.min(x), np.max(x), 0.01)
    axs[i-2].scatter(x[(y==1)], np.zeros(len(x[(y==1)].tolist())), c='red', marker='+')
    axs[i-2].scatter(x[(y==2)], np.zeros(len(x[(y==2)].tolist())), c='green', marker='o')
    axs[i-2].scatter(x[(y==3)], np.zeros(len(x[(y==3)].tolist())), c='blue', marker='x')
    axs[i-2].plot(grid, mix_pdf(grid, mu, sigma, pi), label="GMM")
    axs[i-2].set_title("Density Curves (k=" + str(i) + ")")
    axs[i-2].set_xlabel("Data")
    axs[i-2].set_ylabel("Density")
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