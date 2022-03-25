#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[9]:


df = pd.read_csv('wine_dataset.csv')
df.head()


# In[10]:


df['style'] = df['style'].replace('red',0)


# In[12]:


df['style'] = df['style'].replace('white',1)


# In[13]:


df.head()


# In[14]:


X = df.drop('style',axis=1)
y = df['style']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


# In[17]:


from sklearn.ensemble import ExtraTreesClassifier
modelo = ExtraTreesClassifier()
modelo.fit(X_train,y_train)
result = modelo.score(X_test,y_test)
print('Precisão:',result )


# In[23]:


print(y_test[0:5])
previsao = modelo.predict(X_test[0:5])
print(previsao)


# In[ ]:




