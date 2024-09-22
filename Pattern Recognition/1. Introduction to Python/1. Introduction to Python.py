A = [10, 5, 3, 10, -2, 5, -50]
print(A)
print(A[0])
print(A[0:2])



B = []
for index in [1, 3, 4, 5]:
    B.append(A[index]) # Get indexes from A
print(B)



B = [A[index] for index in [1, 3, 4, 5]]
print(B)



B = [el for el in A if el > 5] # Get the elements of A that are bigger than 5
indices = [i for i, el in enumerate(A) if el > 5] # Get the indexes of A, the elements of which are bigger than 5
print(len(B))



A = [10, 5, 3, 100, -2, 5, -50]
B = [1, 2, 5, 6, 9, 0, 100]
print(A + B) # Create a new list with both of the initial lists
A.extend(B) # Extend the first list with the elements of the second list
print(A)



import pandas as pd
n = [2, 3, 5]
s = ["aa", "bb", "cc"]
b = [True, False, True]
df = pd.DataFrame(list(zip(n, s, b)), columns=['n', 's', 'b']) # Create a dataframe from the three lists and name its columns
print(df)
print(df.n) # Access a specific column of the dataframe
print(df.loc[1]) # Access a specific row of the dataframe
print(df.loc[1, 'b']) # Access a specific row and column of the dataframe
print(df.loc[:, 'b']) # Access all rows of a specific column of the dataframe
print(df.loc[(df.b == True)]) # Access all columns of the rows that satisfy a condition



people = pd.read_csv("./people.txt") # Read a csv file
people = pd.read_csv("./people.txt", sep=";", header=None, names=["Age", "Height", "Weight"]) # Read a csv file with different options

# Get dataframe info
print(people.head())
print(people.describe())
print(people.isnull().sum())

people = people.fillna(people.mean()) # Fill missing values



for i in range(1, 11):
    print(i)

for i in range(0, 101):
    if i % 2 == 0:
        print(i)



from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore", sparse=False) # Create a One-Hot Encoder
encoder.fit(df) # Fit the encoder into the data
df_one_hot = encoder.transform(df) # Transform data to One-Hot Encoding
 