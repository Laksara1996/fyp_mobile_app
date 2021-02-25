# import the libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# importing the dataset with pandas
dataset = pd.read_csv("datasettt.csv")
df = dataset.iloc[:, [0, 1]]

############################# Find clusters values #################################
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans = kmeans.fit(df)
print(kmeans.labels_)

# interpretation of cluster values
l = len(kmeans.labels_)
for i in range(l):
    if kmeans.labels_[i] == 3:
        kmeans.labels_[i] = 1

    elif kmeans.labels_[i] == 0:
        kmeans.labels_[i] = 3

    elif kmeans.labels_[i] == 2:
        kmeans.labels_[i] = 4

    elif kmeans.labels_[i] == 1:
        kmeans.labels_[i] = 2

df['labels'] = kmeans.labels_
print(df['labels'].values.tolist())

############################# Validate the algorithm #################################
# saving the dataframe
df.to_csv('file2.csv', header=False, index=False)

array = df.values
X = array[:, 0:2]
y = array[:, 2]
# split the data set
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=0)

# Make predictions for validating dataset
model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print('Accuracy: ', accuracy_score(Y_validation, predictions))

print('confusion_matrix: ')
print(confusion_matrix(Y_validation, predictions))

print('classification_report: ')
print(classification_report(Y_validation, predictions))

###############check whether the vehicle can reach to the destination or not?################
predict = model.predict([[4, 2]])
distance = 250
litre = 20

if predict == 1:
    rate = 11
elif predict == 2:
    rate = 12.5
elif predict == 3:
    rate = 13.5
elif predict == 4:
    rate = 15

output = rate * litre

if output >= distance:
    print(1)
elif output < distance:
    print(0)
