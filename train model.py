#!/usr/bin/env python
# coding: utf-8

# In[146]:


import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
import joblib as jb


# In[81]:


df = pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")


# In[82]:


df.shape


# In[123]:


df.head(10)


# In[84]:


df.duplicated().sum()


# In[85]:


df.dtypes


# In[135]:


x = df.drop (["addicted_label", "transaction_id", "user_id"], axis=1)
y =df["addicted_label"]


# In[136]:


numerical_cols = x.select_dtypes(include=['int64','float64']).columns.tolist()


# In[137]:


categorical_cols = x.select_dtypes(include=['object']).columns.tolist()


# In[138]:


numerical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])


# In[139]:


categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[140]:


preprocessor = ColumnTransformer(transformers=[
    ('num',numerical_transformer,numerical_cols ),
    ('cat',categorical_transformer,categorical_cols )
])


# In[141]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[142]:


model = Pipeline(steps=[
    ('pre',preprocessor),('reg',LogisticRegression(max_iter=1000))
])


# In[143]:


model.fit(X_train,y_train)


# In[145]:


y_pred = model.predict(X_test)
print(f'{classification_report(y_test,y_pred)}')


# In[147]:


jb.dump(model,'LogisticRegression.pkl')


# In[ ]:




