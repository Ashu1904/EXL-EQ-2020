#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pyxlsb import open_workbook as open_xlsb

df1 = []

with open_xlsb('C:/Users/HP/Desktop/EXL EQ 2020/EXL_EQ_2020_Train_datasets.xlsb') as wb:
    with wb.get_sheet(1) as sheet:
        for row in sheet.rows():
            df1.append([item.v for item in row])

df1 = pd.DataFrame(df1[1:], columns=df1[0])


# In[2]:


import pandas as pd
from pyxlsb import open_workbook as open_xlsb

df2 = []

with open_xlsb('C:/Users/HP/Desktop/EXL EQ 2020/EXL_EQ_2020_Test_Datasets.xlsb') as wb:
    with wb.get_sheet(2) as sheet:
        for row in sheet.rows():
            df2.append([item.v for item in row])

df2 = pd.DataFrame(df2[1:], columns=df2[0])


# In[3]:


df1.head()


# In[4]:


df2.head()


# In[5]:


df1.info()


# In[6]:


df1['var24']=df1['var24'].fillna(value=df1['var24'].mean())


# In[7]:


df1['var24']


# In[8]:


df1


# In[9]:


df1.info()


# In[10]:


df1['var37'].value_counts()


# In[11]:


df1['var37']=df1['var37'].fillna("N")


# In[12]:


df1


# In[13]:


df1.info()


# In[14]:


df1['var36'].value_counts()


# In[15]:


df1['var36']=df1['var36'].fillna("Vedio/Internet")


# In[16]:


df1['var39'].value_counts()


# In[17]:


df1['var39']=df1['var39'].fillna("Single Housing")


# In[18]:


df1


# In[19]:


df1.info()


# In[20]:


from sklearn.preprocessing import LabelEncoder


# In[21]:


le=LabelEncoder()
df1['var33']=le.fit_transform(df1['var33'])
df1['var34']=le.fit_transform(df1['var34'])
df1['var35']=le.fit_transform(df1['var35'])
df1['var40']=le.fit_transform(df1['var40'])
df1['var36']=le.fit_transform(df1['var36'])
df1['var37']=le.fit_transform(df1['var37'])
df1['var39']=le.fit_transform(df1['var39'])


# In[22]:


df1


# In[23]:


df1.info()


# In[24]:


self_service_platform={'Desktop': 1,'Mobile App': 2,'Mobile Web': 3,'STB': 4}


# In[25]:


df1.self_service_platform=[self_service_platform[item] for item in df1.self_service_platform]


# In[26]:


df1


# In[27]:


X=df1.drop(columns=['self_service_platform','var38','var30'])
X


# In[28]:


X.info()


# In[29]:


y=df1.self_service_platform
y


# In[30]:


import statsmodels.api as sm


# In[31]:


x=sm.add_constant(X)
results=sm.OLS(y,x).fit()


# In[32]:


results.summary()


# In[33]:


X=df1.drop(columns=['self_service_platform','var38','var30','var10','var11','var14','var25'])
X


# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[36]:


from sklearn.linear_model import LinearRegression


# In[37]:


lm= LinearRegression()
lm.fit(X_train,y_train)


# In[38]:


print(lm.intercept_)


# In[39]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[40]:


from sklearn.ensemble import RandomForestClassifier


# In[41]:


from sklearn.metrics import classification_report,confusion_matrix


# In[42]:


rfc=RandomForestClassifier(n_estimators=400,criterion='gini',random_state=100)


# In[43]:


rfc.fit(X_train,y_train)


# In[44]:


rfc_pred=rfc.predict(X_test)


# In[45]:


print(confusion_matrix(y_test,rfc_pred))


# In[46]:


print(classification_report(y_test,rfc_pred))


# In[47]:


rfc.score(X_test,y_test)


# In[ ]:


test_data=df2.drop(columns=['var38','var30'])
test_data


# In[116]:


test_data.info()


# In[104]:


test_data['var24']=test_data['var24'].fillna(value=test_data['var24'].mean())


# In[105]:


test_data['var24']


# In[106]:


test_data['var37'].value_counts()


# In[117]:


test_data['var35'].value_counts()


# In[118]:


test_data['var37']=test_data['var37'].fillna("N")
test_data['var35']=test_data['var35'].fillna("Standard")


# In[119]:


test_data['var36']=test_data['var36'].fillna("Vedio/Internet")
test_data['var39']=test_data['var39'].fillna("Single Housing")


# In[120]:


test_data.info()


# In[121]:


le=LabelEncoder()
test_data['var33']=le.fit_transform(test_data['var33'])
test_data['var34']=le.fit_transform(test_data['var34'])
test_data['var35']=le.fit_transform(test_data['var35'])
test_data['var40']=le.fit_transform(test_data['var40'])
test_data['var36']=le.fit_transform(test_data['var36'])
test_data['var37']=le.fit_transform(test_data['var37'])
test_data['var39']=le.fit_transform(test_data['var39'])


# In[122]:


test_data


# In[123]:


test_data.info()


# In[127]:


pred_final=rfc.predict(test_data)


# In[128]:


pred_final


# In[129]:


import pandas as pd


# In[138]:


output=pd.DataFrame({'Customer ID':test_data.cust_id,'Target Platform':pred_final})
output.to_csv(r'C:\Users\HP\Desktop\EXL EQ 2020\My_submission.csv',index=False)

