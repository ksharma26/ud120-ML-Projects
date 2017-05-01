#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

clf10000 = SVC(C = 10000.0,kernel = "rbf")


t_train =time()
clf10000.fit(features_train, labels_train)
print"training time: ", round(time()-t_train,3),"s"
t_prid = time()
prid10000 = clf10000.predict(features_test)
print"predict time: ", round(time()-t_prid,3),"s"
print"Accuracy 10000:", accuracy_score(labels_test, prid10000)
print"Data point 10 ", prid10000[10]
print"26 ", prid10000[26]
print"50 ", prid10000[50]
print"type", type(prid10000)
print"len", len(prid10000)
print"sum", sum(prid10000)
print"alexa", len(prid10000) - sum(prid10000)

#########################################################
### your code goes here ###

#########################################################


