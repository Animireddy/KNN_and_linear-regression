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
# import seaborn as sns; sns.set()
from statistics import stdev,mean
import scipy.stats as ss
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


# In[2]:


admission = pd.read_csv("./AdmissionDataset/data.csv")
admission.head()


# In[3]:


def plotFeatures(col_list,title,label,df):
    plt.figure(figsize=(10, 14))
    i = 0
    print(len(col_list))
    for col in col_list:
        i+=1
        plt.subplot(7,2,i)
        plt.plot(df[col],df[label],marker='.',linestyle='none')
        xlabel(col)
        ylabel(label)
        plt.title(title % (col))   
        plt.tight_layout()


# In[4]:


print(admission.keys())
y_adm = admission.Coa.values
print(len(admission.Coa.unique()))
X_adm = admission.iloc[:,:].drop('Coa', axis = 1)
X_adm = X_adm.drop('Serial', axis = 1)
#X_adm = X_adm.drop('Research', axis = 1)
print(y_adm.shape)
print(X_adm.shape)
X_adm_train,X_adm_val,y_adm_train,y_adm_val=train_test_split(X_adm,y_adm,test_size=0.2,random_state=42)


X_adm_train=X_adm_train.values
X_adm_val=X_adm_val.values


# In[24]:


colnames = list(admission.keys())
plotFeatures(colnames,"Relationship bw %s and output", 'Coa',admission)


# In[25]:


def mean_square(train, l_rate, n_epoch,Class):
    #print(type(train[0].values))
    m=train.shape[0]
    w = np.zeros((train.shape[1],1))
    b = 0
    v= np.ones((m,1))
    mean = np.mean(train,0)
    std = np.std(train,0)
    train=(train-mean)/std;
    for i in range(n_epoch):
        Class = Class.reshape((-1,1))
        yhat= (np.dot(train,w) + b*v).reshape((-1,1))
        J = (1/(2*m))*(np.dot(np.transpose(yhat-Class),yhat-Class))
        dw =  np.dot(np.transpose(train),yhat-Class)
        db =  np.dot(np.transpose(v),yhat-Class)
        w = w - l_rate*dw
        b = b - l_rate*db[0][0]
    
    
    #print("Error Value",J)
    
    return w,b


# In[26]:


def linear_regression(train_data,l_rate, n_epoch,Class,inp,function):
    if(function == 'MSE'):
         w,b = mean_square(train_data, l_rate, n_epoch,Class)
    elif(function == 'MAE'):
         coef = mean_absolute(train_data, l_rate, n_epoch,Class)
    elif(function == 'MAPE'):
         coef = MAPE(train_data,l_rate, n_epoch, n_epoch,Class)
    
    mean = np.mean(train_data,0)
    std = np.std(train_data)
    inp = (inp-mean)/std;
    prob=inp.dot(inp,w)+b
    return prob     


# In[16]:



value = linear_regression(X_adm_train,0.0007,1000,y_adm_train,X_adm_val,'MSE')
print("prob",value)


# ### Mean square error loss function

# In[30]:


score = 0
for ind in range(0,X_adm_val.shape[0]):
    error = value[ind] - y_adm_val[ind]
    score += error ** 2
print(score)


# ### Mean Absolute error function 

# In[31]:


score1 = 0
for ind in range(0,X_adm_val.shape[0]):
    error = value[ind] - y_adm_val[ind]
    score1 += abs(error)
print(score1)


# ### Mean absolute percentage error function

# In[32]:


score2 = 0
for ind in range(0,X_adm_val.shape[0]):
    error = y_adm_val[ind] - value[ind]
    score2 += float(error)/float(y_adm_val[ind])
print(score2)


# In[ ]:





# In[ ]:





# In[ ]:




