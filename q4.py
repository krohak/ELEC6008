
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
#print("Features: ",features)


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

#print("Xtrain: ", Xtrain)
#print(features)


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

Xtest = []


# In[12]:

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


# In[13]:

Xtest = np.asarray(Xtest)
Xtest.shape


# # Q4

# In[14]:

import matplotlib.pyplot as plt
import matplotlib.colors
from cvxopt import solvers
from cvxopt import matrix


# ## Two Features

print("--------------------TWO FEATURES---------------")

# In[15]:

# choose only "car" and "hotel" features for the train data
Xtrain_two = np.concatenate((Xtrain[:20, [0,7]], -Xtrain[20:, [0, 7]]))

Y_two = np.append( np.ones((1, 20)), -np.ones((1, 20)))
Y_two = Y_two.reshape(40,1)
Y_two = np.concatenate((Y_two,Xtrain_two), axis=1)

Y_two.shape


# In[16]:

# choosing "car" and "hotel" features only for the test data
Xtest_two = np.concatenate((np.ones((10, 1)),Xtest[:, [0,7]]), axis=1)


# ### Using Perceptron

print("--------------------PERCEPTRON---------------")

# In[17]:

a=np.zeros((Y_two.shape[1],1))

# no. of misclassified samples
sum_wrong=1

#Perceptron
a_iter=a
k=0

while sum_wrong>0 and k<1000:
    
    wrong=np.dot(Y_two,a_iter)<=0
    sum_wrong=sum(wrong)
    sum1=sum(wrong*np.ones((1,Y_two.shape[1]))*Y_two)    
    a_iter=a_iter+sum1.reshape(Y_two.shape[1],1)
    
    k=k+1

print("Final a = {} after k={} iterations".format(np.transpose(a_iter), k, ))


# #### Perceptron does not converge with two features since this dataset contains an observation 0,0 [car, hotel] which belongs to both classes. This means that there is data with neither of the keywords "car" and "hotel" which belong to both the classes, and it is difficult for the perceptron, a linear classifier, to classify the data. Thus, the classifier cannot be obtained

# ### Using SVM

print("--------------------SVM----------------------")

# In[18]:

A=matrix(Y_two,tc='d')
b=matrix(-1*np.ones((A.size[0],1)),tc='d')

q1=np.zeros((1,A.size[1]))
Q2=np.concatenate((np.zeros((Xtrain_two.shape[1],1)), np.eye(Xtrain_two.shape[1])),axis=1)
Q=np.concatenate((q1,Q2),axis=0)
Q=matrix(2*Q,tc='d')

q=matrix(np.zeros((A.size[1],1)),tc='d')


# In[19]:

#solvers.options['show_progress'] = False
sol=solvers.qp(Q,q,A,b)


# #### SVM does not converge with two features since this dataset containains an observation 0,0 [car, hotel] which belongs to both classes. This means that there is data with neither of the keywords "car" and "hotel" which belong to both the classes, and it is difficult for the linear version of SVM to classify the data. Thus, we cannot obtain the classifier.

# ### Testing accuracy

print("--------------------ACCURACY----------------------")

# In[20]:

Y_true = [0,1,1,0,0,0,1,1,0,1]

for i, val in enumerate(Y_true):
    if val == 0:
        Y_true[i] = -1


# In[21]:

#Perceptron classifier
a_con_p=a_iter
ans = np.dot(np.transpose(a_con_p), np.transpose(Xtest_two))
match = sum([1 for i in range(len(Y_true)) if Y_true[i]*ans[0][i] < 0])
print("Accuracy of Perceptron with 2 features is equal to {} percent".format(match/len(Y_true)*100))
print("Note: the classifier did not converge")


# In[22]:

#SVM classifier
a_con_s=sol['x']
ans = np.dot(np.transpose(a_con_s), np.transpose(Xtest_two))
match = sum([1 for i in range(len(Y_true)) if Y_true[i]*ans[0][i] > 0])
print("Accuracy of SVM with 2 features is equal to {} percent".format(match/len(Y_true)*100))
print("Note: the classifier did not converge")


print("--------------------------------------------")


# ## All Features

print("--------------------ALL FEATURES--------------")


# In[23]:

# choose all features for the train data
Xtrain_all = np.concatenate((Xtrain[:20], -Xtrain[20:]))

Y_all = np.append( np.ones((1, 20)), -np.ones((1, 20)))
Y_all = Y_all.reshape(40,1)
Y_all = np.concatenate((Y_all,Xtrain_all), axis=1)

Y_all.shape


# In[24]:

# choosing all features for the test data
Xtest_all = np.concatenate((np.ones((10, 1)),Xtest), axis=1)

Xtest_all.shape


# ### Using Perceptron

print("--------------------PERCEPTRON---------------")

# In[25]:

a=np.zeros((Y_all.shape[1],1))

# no. of misclassified samples
sum_wrong=1

#Perceptron
a_iter=a
k=0

while sum_wrong>0 and k<1000:
    
    wrong=np.dot(Y_all,a_iter)<=0
    sum_wrong=sum(wrong)
    sum1=sum(wrong*np.ones((1,Y_all.shape[1]))*Y_all)    
    a_iter=a_iter+sum1.reshape(Y_all.shape[1],1)
    
    k=k+1

print("Final a = {} after k={} iterations".format(np.transpose(a_iter), k, ))


# ### Using SVM

print("--------------------SVM----------------------")


# In[26]:

A=matrix(Y_all,tc='d')
b=matrix(-1*np.ones((A.size[0],1)),tc='d')

q1=np.zeros((1,A.size[1]))
Q2=np.concatenate((np.zeros((Xtrain_all.shape[1],1)), np.eye(Xtrain_all.shape[1])),axis=1)
Q=np.concatenate((q1,Q2),axis=0)
Q=matrix(2*Q,tc='d')

q=matrix(np.zeros((A.size[1],1)),tc='d')


# In[27]:

solvers.options['show_progress'] = True
sol=solvers.qp(Q,q,A,b)


# ### Testing accuracy
print("--------------------ACCURACY----------------------")

# In[28]:

#Perceptron classifier
a_con_p=a_iter
ans = np.dot(np.transpose(a_con_p), np.transpose(Xtest_all))
match = sum([1 for i in range(len(Y_true)) if Y_true[i]*ans[0][i] < 0])
print("Accuracy of Perceptron with all features is equal to {} percent".format(match/len(Y_true)*100))
print("Note: the classifier converged after 3 iterations")


# In[29]:

#SVM classifier
a_con_s=sol['x']
ans = np.dot(np.transpose(a_con_s), np.transpose(Xtest_all))
match = sum([1 for i in range(len(Y_true)) if Y_true[i]*ans[0][i] > 0])
print("Accuracy of SVM with all features is equal to {} percent".format(match/len(Y_true)*100))
print("Note: the classifier converged after 8 iterations")

print("------------------------------------------")
