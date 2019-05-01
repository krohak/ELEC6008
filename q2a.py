
# coding: utf-8

# In[1]:

import glob
import re


# In[2]:

fcf = glob.glob('./*.txt')


# In[3]:

f = open('01.txt','r')
text1 = f.read()


# In[4]:

f = open('feature.txt','r')
features = f.read()
features = re.split(r'[` \n]', features)
print(features)

# In[5]:

feature1 = features[0]
print(feature1)


# In[6]:

occurance = 0
for word in re.split(r'[` \-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', text1):
    if word == feature1: occurance += 1 


# In[7]:

print(occurance)

