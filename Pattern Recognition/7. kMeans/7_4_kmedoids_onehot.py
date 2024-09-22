import pandas as pd

from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import OneHotEncoder

# import data
Rank = ["High", "Low", "High", "Low", "Low", "High"]
Topic = ["SE", "SE", "ML", "DM", "ML", "SE"]
conferences = pd.DataFrame({"Rank": Rank, "Topic": Topic})

# transform data
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoder.fit(conferences)
conferences = encoder.transform(conferences)

# apply K-Medoids
kmedoids = KMedoids(n_clusters=3, method='pam').fit(conferences)
print('centers ', kmedoids.cluster_centers_)
print('labels ', kmedoids.labels_)
conferences = pd.DataFrame(conferences)
print(conferences.loc[kmedoids.medoid_indices_, :])