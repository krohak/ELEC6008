
# coding: utf-8

# In[1]:

import glob
import re
import numpy as np


# In[2]:

fcf = glob.glob('./*.txt')


# In[3]:

f = open('01.txt','r')
text1 = f.read()


# In[4]:

f = open('feature.txt','r')
features = f.read()
features = re.split(r'[` \n]', features)
print("Features: ",features)


# In[5]:

Xtrain = []


# In[6]:

file_features = []
for feature in features:
    occurance = 0
    for word in re.split(r'[` \-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', text1):
        if word == feature: occurance += 1
    file_features.append(occurance)

Xtrain.append(file_features)


# In[7]:

print("Xtrain: ", Xtrain)
print(features)


# In[8]:

Xtrain = []


# In[9]:

for file_number in range(1, 41):
    
    file_name = "{:02d}.txt".format(file_number)
    f = open(file_name,'r')
    text = f.read()
    
    file_features = []
    for feature in features:
        occurance = 0
        for word in re.split(r'[` \-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', text):
            if word == feature: occurance += 1 
        file_features.append(occurance)
        
    Xtrain.append(file_features)


# In[10]:

Xtrain = np.asarray(Xtrain)
Xtrain.shape


# In[11]:

print("Xtrain: ", Xtrain)


# In[12]:

Xtest = []


# In[13]:

for file_number in range(41, 51):
    
    file_name = "{:02d}.txt".format(file_number)
    f = open(file_name,'r')
    text = f.read()
    
    file_features = []
    for feature in features:
        occurance = 0
        for word in re.split(r'[` \-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', text):
            if word == feature: occurance += 1 
        file_features.append(occurance)
        
    Xtest.append(file_features)


# In[14]:

Xtest = np.asarray(Xtest)
Xtest.shape


# In[15]:

print("Xtest: ", Xtest)


# ## Q3

# In[16]:

import matplotlib.pyplot as plt
import matplotlib.colors
from cvxopt import solvers
from cvxopt import matrix


# In[17]:

# choose only "car" and "hotel" features
Xtrain = np.concatenate((Xtrain[:20, [0,7]], -Xtrain[20:, [0, 7]]))
Xtrain.shape


# In[18]:

Y = np.append( np.ones((1, 20)), -np.ones((1, 20)))
Y = Y.reshape(40,1)
Y = np.concatenate((Y,Xtrain), axis=1)


# In[19]:

Y.shape


# ## Using Perceptron

# In[20]:

a=np.zeros((Y.shape[1],1))

# no. of misclassified samples
sum_wrong=1

#Perceptron
a_iter=a
k=0

while sum_wrong>0 and k<1000:
    
    wrong=np.dot(Y,a_iter)<=0
    sum_wrong=sum(wrong)
#     print("sum wrong",sum_wrong)
    sum1=sum(wrong*np.ones((1,Y.shape[1]))*Y)    
    a_iter=a_iter+sum1.reshape(Y.shape[1],1)
    
    k=k+1

print("Final a = {} after k={} iterations".format(np.transpose(a_iter), k, ))


# #### Perceptron does not converge with two features since this dataset containains an observation 0,0 [car, hotel] which belongs to both classes. This means that there is data with neither of the keywords "car" and "hotel" which belong to both the classes, and it is difficult for the perceptron, a linear classifier, to classify the data. Thus, the classifier cannot be obtained

# ## Using SVM

# In[21]:

A=matrix(Y,tc='d')
b=matrix(-1*np.ones((A.size[0],1)),tc='d')

q1=np.zeros((1,A.size[1]))
Q2=np.concatenate((np.zeros((Xtrain.shape[1],1)), np.eye(Xtrain.shape[1])),axis=1)
Q=np.concatenate((q1,Q2),axis=0)
Q=matrix(2*Q,tc='d')

q=matrix(np.zeros((A.size[1],1)),tc='d')


# In[22]:

sol=solvers.qp(Q,q,A,b)


# #### SVM does not converge with two features since this dataset containains an observation 0,0 [car, hotel] which belongs to both classes. This means that there is data with neither of the keywords "car" and "hotel" which belong to both the classes, and it is difficult for the linear version of SVM to classify the data. Thus, we cannot obtain the classifier.

# ## Plotting the Data and Classifiers

# In[23]:

#Perceptron classifier
a_con=a_iter
x=np.arange(-1,100,15)
y1 = -(a_con[0]+sum(a_con[1:-1]*x))/a_con[-1]
y1


# In[24]:

#SVM classifier
a_con=sol['x']
y2=-(a_con[0]+a_con[1]*x)/a_con[2]
y2


# In[25]:

axes = plt.gca()

axes.set_xlim([-1,8])
axes.set_ylim([-1,8])
plt.xlabel("Car")
plt.ylabel("Hotel")


plt.scatter(Xtrain[0:20,0], Xtrain[0:20,1], color='k', label="Class 1: Car")
plt.scatter(Xtrain[20:,0], -Xtrain[20:,1], color='r', label= "Class 2: Hotel")

# perceptron classifier
plt.plot(x,y1, label="Perceptron")

plt.legend()
plt.show()


# In[27]:

axes = plt.gca()

axes.set_xlim([-1,8])
axes.set_ylim([-1,8])
plt.xlabel("Car")
plt.ylabel("Hotel")


plt.scatter(Xtrain[0:20,0], Xtrain[0:20,1], color='k', label="Class 1: Car")
plt.scatter(Xtrain[20:,0], -Xtrain[20:,1], color='r', label= "Class 2: Hotel")

# svm classifier
plt.plot(x,y2, c="orangered",label="SVM")

plt.legend()
plt.show()

