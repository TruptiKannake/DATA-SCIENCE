#!/usr/bin/env python
# coding: utf-8

# # TASK  1: Iris Flowers Classfication ML Project.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data= pd.read_csv(r"D:\LGM\iris.data")


# In[3]:


data.head()


# In[4]:


data.columns= ['Sepal_Length','Sepal_Width','Petal_Length','Petal_width','Class']


# In[5]:


data.head()


# # CHECKING  THE DATA TYPES IN DATA SET

# In[6]:


data.info()


# In[7]:


data.describe()


# # CHECKING FOR MISSING  VALUE

# In[8]:


data.isnull().sum()


# # DATA VISUALIZATION

# In[9]:


plt.bar(data['Class'],data['Sepal_Length'],width=0.5)
plt.title('Sepal_Length vs Type')
plt.show()


# In[10]:


plt.bar(data['Class'],data['Petal_Length'], width=0.5)
plt.title('Petal_Length vs Type')
plt.show()


# In[11]:


sns.pairplot(data,hue = 'Class')


# # SPLITTING THE DATA

# In[12]:


from sklearn.model_selection import train_test_split


# In[14]:


x= data.drop ('Class', axis=1)
y = data['Class']
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.33, random_state=42)


# In[15]:


from sklearn.linear_model import LogisticRegression


# In[16]:


model= LogisticRegression()


# In[18]:


model.fit(x_train,y_train)


# In[19]:


pred = model.predict(x_test)
pred


# In[20]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[21]:


print(accuracy_score(y_test,pred)*100)


# In[22]:


print(confusion_matrix(y_test,pred))


# In[23]:


print (classification_report(y_test,pred))

