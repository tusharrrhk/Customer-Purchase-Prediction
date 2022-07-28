#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv('Social_Network_Ads.csv')


# In[3]:


dataset


# In[5]:


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# In[36]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[37]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=101)
classifier.fit(X_train,y_train)
predictions = classifier.predict(X_test)


# In[38]:


predictions


# In[39]:


y_test


# In[40]:


print(classifier.predict(sc.transform([[30,87000]])))


# In[41]:


print(np.concatenate((predictions.reshape(len(predictions),1),y_test.reshape(len(y_test),1)),1))


# In[42]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(predictions,y_test))
print('\n')
print(classification_report(predictions,y_test))
print('\n')
print(accuracy_score(predictions,y_test))


# In[ ]:





# In[ ]:




