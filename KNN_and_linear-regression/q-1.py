#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os
import math
import operator
from pylab import *
import matplotlib
import matplotlib.pyplot as plt
import sklearn.neighbors as sn
from statistics import stdev,mean
import scipy.stats as ss
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


# #### KNN Classifier

# In[12]:


def KNN_classifier(k,vari, dataset,Class,distancemeasure):
    if distancemeasure == 'euclidean':
        distance = []
        for x in range(0,dataset.shape[0]):
            vari2 = list(dataset[x][:])
            l = len(vari)
            distanceance = 0
            for i in range(0,l):
                distanceance += pow((float(vari[i]) - float(vari2[i])), 2)
            distance.append((Class[x],math.sqrt(distanceance)))
    
    elif distancemeasure == 'manhattan':
        distance = []
        for x in range(0,dataset.shape[0]):
            vari2 = list(dataset[x][:])
            l = len(vari)
            distanceance = 0
            for i in range(0,l):
                distanceance += abs(vari[i]-vari2[i])
            distance.append((Class[x],distanceance))
    
    elif distancemeasure == 'chebychev':
        distance = []
        for x in range(0,dataset.shape[0]):
            vari2 = list(dataset[x][:])
            l = len(list1)
            distanceance = 0
            for i in range(0,l):
                distanceance = max(distanceance,abs(list1[i]-list2[i]))
            distance.append((Class[x],distanceance))
            
    elif distancemeasure == 'cosine':
        distance = []
        for x in range(0,dataset.shape[0]):
            vari2 = list(dataset[x][:])
            distance.append((Class[x],cosine_measure(vari,vari2)))
            
    
    distance.sort(key=operator.itemgetter(1))
    #print(distance)
    uniq = {}
    count = 0
    for y in distance:
        count += 1
        if y[0] not in uniq.keys():
            uniq[y[0]] = 1
        else:
            uniq[y[0]]+=1
        if count == k:
            break
    
    maxim = 0
    for key in uniq.keys():
        if maxim < uniq[key]:
            maxim = uniq[key]
            Class = key
    
    return Class
        


# In[3]:


def printvals(train_data,train_class,val_data,val_class,measure,K):
    pred = []
    Mean = np.mean(train_data,0)
    Std = np.std(train_data,0)
    temp = []
    for i in range(0,train_data.shape[0]):
        vari = train_data.iloc[i,:].values
        vari1 = (vari-Mean)/(Std)
        temp.append(vari1)
    temp = np.array(temp)
        
    #print("hello"," ",type(train_dataS))
    for i in range(0,val_data.shape[0]):
        vari = val_data.iloc[i,:].values
        vari2 = (vari-Mean)/(Std)
        pred.append(KNN_classifier(K,list(vari2), temp,train_class ,measure))

    print("F1-score",f1_score(val_class,pred, average = 'micro'))
    print("precision",precision_score(val_class,pred, average = 'micro'))
    print("Accuracy",accuracy_score(val_class,pred))
    print("Recall",recall_score(val_class,pred, average = 'micro'))
    return accuracy_score(val_class,pred)


# In[4]:


def printvals_inbuilt(train_data,train_class,val_data,val_class,measure,K):
    neigh = sn.KNeighborsClassifier(n_neighbors = K, weights = 'uniform', algorithm = 'auto', leaf_size = 30, p = 2, metric = 'minkowski')
    train_dataS1=scale(train_data)
    neigh.fit(train_dataS1,train_class)
    val_dataS=scale(val_data)
    pred = neigh.predict(val_dataS)
    print("F1-score",f1_score(val_class,pred, average = 'micro'))
    print("precision",precision_score(val_class,pred, average = 'micro'))
    print("Accuracy",accuracy_score(val_class,pred))
    print("Recall",recall_score(val_class,pred, average = 'micro'))


# In[5]:


iris = pd.read_csv("./Iris/Iris.csv")
data1 = pd.read_csv("./RobotDataset/Robot1.csv")
data2 = pd.read_csv("./RobotDataset/Robot2.csv")


# ### PART - 1

# In[6]:


y_iris = iris.Class.values
X_iris = iris.iloc[:,:].drop('Class', axis = 1)
X_iris_train,X_iris_val,y_iris_train,y_iris_val=train_test_split(X_iris,y_iris,test_size=0.2,stratify=y_iris,random_state=42)
printvals(X_iris_train,y_iris_train,X_iris_val,y_iris_val,'euclidean',20)
print("\ninbult KNN classifier")
printvals_inbuilt(X_iris_train,y_iris_train,X_iris_val,y_iris_val,'euclidean',20)


# In[7]:


y_data1 = data1.Class.values
X_data1 = data1.iloc[:,:].drop('Class', axis = 1)
X_data1 = X_data1.drop('Id', axis = 1)
print(y_data1.shape)
print(X_data1.shape)
X_data1_train,X_data1_val,y_data1_train,y_data1_val=train_test_split(X_data1,y_data1,test_size=0.2,stratify=y_data1,random_state=42)

printvals(X_data1_train,y_data1_train,X_data1_val,y_data1_val,'euclidean',10)
print("\ninbult KNN classifier")
printvals_inbuilt(X_data1_train,y_data1_train,X_data1_val,y_data1_val,'euclidean',20)


# In[8]:


y_data2 = data2.Class.values
X_data2 = data2.iloc[:,:].drop('Class', axis = 1)
X_data2 = X_data2.drop('Id', axis = 1)
X_data2_train,X_data2_val,y_data2_train,y_data2_val=train_test_split(X_data2,y_data2,test_size=0.2,stratify=y_data2,random_state=42)

printvals(X_data2_train,y_data2_train,X_data2_val,y_data2_val,'euclidean',10)
print("\ninbult KNN classifier")
printvals_inbuilt(X_data2_train,y_data2_train,X_data2_val,y_data2_val,'euclidean',10)


# ### PART - 2

# In[9]:


Accuracy = []
K_parameter = []
for k in range(5,30):
    K_parameter.append(k)
    Accuracy.append(printvals(X_iris_train,y_iris_train,X_iris_val,y_iris_val,'euclidean',k))

figure()
plot(K_parameter, Accuracy, 'r')
xlabel('K_parameter')
ylabel('Accuracy')
title('iris')
show()


# In[10]:


Accuracy = []
K_parameter = []
for k in range(5,30):
    K_parameter.append(k)
    Accuracy.append(printvals(X_data1_train,y_data1_train,X_data1_val,y_data1_val,'euclidean',k))

figure()
plot(K_parameter, Accuracy, 'r')
xlabel('K_parameter')
ylabel('Accuracy')
title('data1')
show()


# In[11]:


Accuracy = []
K_parameter = []
for k in range(5,30):
    K_parameter.append(k)
    Accuracy.append(printvals(X_data2_train,y_data2_train,X_data2_val,y_data2_val,'euclidean',k))

figure()
plot(K_parameter, Accuracy, 'r')
xlabel('K_parameter')
ylabel('Accuracy')
title('data2')
show()




