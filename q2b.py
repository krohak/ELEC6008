
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


# #### 2. Based on the results from Q2a, count all the occurrence of each keyword of “features” in “01.txt” and store the results in the first row of a matrix called Xtrain

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


# #### 3. Repeat for “01.txt” to “40.txt”. For “xx.txt”, store the occurrence of each keywords in the xx-th row of the matrix Xtrain

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


# ####  4. For “41.txt” to “50.txt”, save the occurrence of each keywords to the matrix Xtest. (You should map “41.txt” to the 1 st row of the matix)

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


# #### 5. Create a vector Y that stores the label: Class 0 for 01.txt to 20.txt Class 1 for 21.txt to 40.txt

# In[16]:

Y = np.append( np.zeros((1, 20)), np.ones((1, 20)))
Y = Y.reshape(1,40)


# In[17]:

print("Y: ", Y)

