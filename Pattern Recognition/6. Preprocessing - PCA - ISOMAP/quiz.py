import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score
import matplotlib.pyplot as plt

data = pd.read_csv("./quiz_data.csv")

trainingRange = list(range(0,50))+list(range(90,146))
training = data.loc[trainingRange, :]
trainingType = training.loc[:, 'Type']
training = training.drop(["Type"], axis=1)

testingRange = list(range(50,90))
testing = data.loc[testingRange, :]
testingType = testing.loc[:, 'Type']
testing = testing.drop(["Type"], axis=1)

scaler = StandardScaler()
scaler = scaler.fit(training)
training_scaled = scaler.transform(training)
pca = PCA()
pca.fit(training_scaled)
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_
info_gain = (eigenvalues[0]) / sum(eigenvalues)
print(info_gain)
#####################
info_loss = 1-(sum(eigenvalues[0:4])) / sum(eigenvalues)
print(info_loss)
####################
testing_scaled = scaler.transform(testing)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(training, trainingType)
pred = clf.predict(testing)
print(accuracy_score(testingType, pred))
######################
print(recall_score(testingType, pred, pos_label=2))
#####################
training_transformed = pca.transform(training_scaled)
print(len(training_transformed[:,0:0]))
for i in range(1,len(training_transformed[0,:])+1):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(training_transformed[:,0:i], trainingType)
    testing_transformed = pca.transform(testing_scaled)
    pred = clf.predict(testing_transformed[:,0:i])
    print(accuracy_score(testingType, pred))

# scaler = StandardScaler()
# scaler = scaler.fit(training)
# training_scaled = scaler.transform(training)
# testing_scaled = scaler.transform(testing)

# pca = PCA()
# pca = pca.fit(training_scaled)
# training_pca = pca.transform(training_scaled)
# testing_pca = pca.transform(testing_scaled)

# eigenvalues = pca.explained_variance_
# # print(1-(sum(eigenvalues[0:4])/sum(eigenvalues)))
# # plt.bar(range(len(eigenvalues)), eigenvalues/sum(eigenvalues))
# # plt.show()

# print(len(training_pca[0,:]))
# for i in range(1,len(training_pca[0,:])):

#     clf = KNeighborsClassifier(n_neighbors=3)
#     clf = clf.fit(training_pca[:,0:i], trainingType)

#     pred = clf.predict(testing_pca[:,0:i])
#     print(i, accuracy_score(testingType, pred))
# # print(recall_score(testingType, pred, pos_label=2))