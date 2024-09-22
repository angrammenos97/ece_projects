from sklearn import datasets
import statistics

iris = datasets.load_iris()

print(statistics.mean(iris.data[:,iris.feature_names.index("petal length (cm)")]))
print(max(iris.data[:,iris.feature_names.index("sepal width (cm)")]))
print(statistics.variance(iris.data[:,iris.feature_names.index("sepal length (cm)")]))

means = []
for i in range(0,len(iris.feature_names)):
    means.append(statistics.mean(iris.data[:,i]))

print(means)
