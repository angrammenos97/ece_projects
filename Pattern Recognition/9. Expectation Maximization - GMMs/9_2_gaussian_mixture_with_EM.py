import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

from sklearn.mixture import GaussianMixture


# import data
gdata = pd.read_csv("./gdata.txt")
x = gdata.loc[:, "X"]
y = gdata.loc[:, "Y"]

# question1 plot
# plt.scatter(x[(y==1)], np.zeros(len(x[(y==1)].tolist())), c='red', marker='+')
# plt.scatter(x[(y==2)], np.zeros(len(x[(y==2)].tolist())), c='blue', marker='o')
# sns.histplot(x, kde=True, stat="density", linewidth=0, fill=False)
# plt.show()

# question3 apply EM
mu = [0, 1]
lamda = [0.5, 0.5]
epsilon = 1e-8
log_likelihood = np.sum([math.log(i) for i in lamda[0]*norm.pdf(x, loc=mu[0], scale=1)+lamda[1]*norm.pdf(x, loc=mu[1], scale=1)])
# Loop until convergence
while True:
    # Expectation step
    # Find distributions given mu, lamda (and sigma)
    T1 = norm.pdf(x, loc=mu[0], scale=1)
    T2 = norm.pdf(x, loc=mu[1], scale=1)
    P1 = lamda[0] * T1 / (lamda[0] * T1 + lamda[1] * T2)
    P2 = lamda[1] * T2 / (lamda[0] * T1 + lamda[1] * T2)
    # Maximization step
    # Find mu, lamda (and sigma) given the distributions
    mu[0] = np.sum(P1 * x) / np.sum(P1)
    mu[1] = np.sum(P2 * x) / np.sum(P2)
    lamda[0] = np.mean(P1)
    lamda[1] = np.mean(P2)
    # Calculate the new log likehood (to be maximized)
    new_log_likelihood = np.sum([math.log(i) for i in lamda[0]*norm.pdf(x, loc=mu[0], scale=1)+lamda[1]*norm.pdf(x, loc=mu[1], scale=1)])
    # Print the current paramters and the likehood
    print("mu=", mu, "\tlamba=", lamda, "\tlog_likehood=", new_log_likelihood)
    # Break if the algorithm converges
    if (new_log_likelihood - log_likelihood <= epsilon) : break
    log_likelihood = new_log_likelihood

# question4 apply Gaussian Mixture
data = np.array(x.tolist()).reshape(-1,1)
gm = GaussianMixture(n_components=2).fit(data)
print("Means ", gm.means_)
print("Covariances ", gm.covariances_)
print("Logarithmic Propability Surface ", np.sum(gm.score_samples(data)))
def mix_pdf(x, loc, scale, weights):
    d = np.zeros_like(x)
    for mu, sigma, pi in zip(loc, scale, weights):
        d += pi * norm.pdf(x, loc=mu, scale=sigma)
    return d
pi, mu, sigma = gm.weights_.flatten(), gm.means_.flatten(), np.sqrt(gm.covariances_.flatten())
grid = np.arange(np.min(x), np.max(x), 0.01)
plt.scatter(x[(y==1)], np.zeros(len(x[(y==1)].tolist())), c='red', marker='+')
plt.scatter(x[(y==2)], np.zeros(len(x[(y==2)].tolist())), c='blue', marker='o')
plt.plot(grid, mix_pdf(grid, mu, sigma, pi), label="GMM")
sns.histplot(x, kde=True, stat="density", linewidth=0, fill=False)
plt.legend(loc='upper right')
plt.show()