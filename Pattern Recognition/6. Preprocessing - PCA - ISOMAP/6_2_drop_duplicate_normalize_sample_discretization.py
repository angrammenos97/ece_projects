import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


# import data
engdata = pd.read_csv("./engdata.txt")
pdata = engdata.loc[:, ["Age", "Salary"]]

# question1 delete duplicates and normalize
pdata.drop_duplicates()
scaler = StandardScaler()
scaler = scaler.fit(pdata)
transformed = pd.DataFrame(scaler.transform(pdata), columns=["Age", "Salary"])
# plt.scatter(pdata.Age, pdata.Salary)
# plt.show()
# plt.scatter(transformed.Age, transformed.Salary)
# plt.show()

# question2 sample data
data_sample = pdata.sample(n=150, random_state=1, replace=True)
# plt.scatter(pdata.Age, pdata.Salary)
# plt.show()
# plt.scatter(data_sample.Age, data_sample.Salary)
# plt.show()

# question3 discretization
discAge = pd.cut(pdata.Age, [0, 10, 20, 30, 40, 50, 60, 70, 80])
discSalary = pd.cut(pdata.Salary, pd.interval_range(start=0, freq=400, end=4000))
