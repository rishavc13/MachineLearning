#!/usr/bin/python3

from sklearn import tree
# apple and orange -- texture and weight
# smooth=0   and    plumpy=1
features=[[110,0],[120,0],[130,1],[140,1]]

output=["Apple","Apple","Orange","Orange"]

# now loading decision tree classifier
algo=tree.DecisionTreeClassifier()

# now training the features and output set
trained=algo.fit(features,output)     # generally 1 free core of cpu is required

# testing trained model  ----   Q & A
output1=trained.predict([[90,0]])

# printing result
print(output1)
