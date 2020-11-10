# -*- coding: utf-8 -*-
"""
@author: vchan
"""

#------------Exercise 1-------------

from sklearn.neighbors import KNeighborsClassifier
import numpy
import pandas
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

risk_df = pandas.read_csv("train.csv")

#Subset the required columns
X = risk_df.iloc[:,2:5]
y = risk_df['risk']

#Standardize the columns
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X_std = standardizer.fit_transform(X)

#Split the train and test rows
X_std_train, X_std_test, y_train, y_test= train_test_split (X_std, y, test_size = 0.33, random_state= 500)

#Build model
from sklearn.neighbors import KNeighborsClassifier
knn2 = KNeighborsClassifier(n_neighbors=3, p=2).fit(X_std_train,y_train)

#Generate prediction
y_test_pred = knn2.predict(X_std_test)

#Generate scores
accuracy = metrics.accuracy_score(y_test, y_test_pred)
precision = metrics.precision_score(y_test, y_test_pred, pos_label='good risk')
recall = metrics.recall_score(y_test, y_test_pred, pos_label='good risk')

#Save scores
with open("metrics.txt", 'w') as outfile:
    outfile.write("Accuracy" % accuracy)
    outfile.write("Precision" % precision)
    outfile.write("Recall" % recall)

#Save confusion matrix

labels = ['good risk', 'bad loss']
cm = metrics.confusion_matrix(y_test, y_test_pred, labels)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')

plt.savefig("confusion_matrix.png",dpi=120) 
plt.close()




