
# coding: utf-8

# In[14]:

import glob
fcf = glob.glob('./*.txt')
print(fcf)

# In[33]:

f = open('01.txt','r')
text1 = f.read()
print(text1)
