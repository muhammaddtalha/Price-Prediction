#!/usr/bin/env python
# coding: utf-8

# # Price Prediction using Regression

# This is a tickets pricing monitoring system. It scrapes tickets pricing data periodically and stores it in a database. Ticket pricing changes based on demand and time, and there can be significant difference in price. We are creating this product mainly with ourselves in mind. Users can set up alarms using an email, choosing an origin and destination (cities), time (date and hour range picker) choosing a price reduction over mean price, etc.

# **Following is the description for columns in the dataset**<br>
# - insert_date: date and time when the price was collected and written in the database<br>
# - origin: origin city <br>
# - destination: destination city <br>
# - start_date: train departure time<br>
# - end_date: train arrival time<br>
# - train_type: train service name<br>
# - price: price<br>
# - train_class: ticket class, tourist, business, etc.<br>
# - fare: ticket fare, round trip, etc <br>

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# #### **Task 1: Import Dataset and create a copy of that dataset**

# In[2]:


data = pd.read_csv('data1.csv')
df = data.copy() 


# #### **Task 2: Display first five rows** 

# In[3]:


df.head(5)


# #### **Task 3: Drop 'unnamed: 0' column**

# In[4]:


df.drop('Unnamed: 0',axis = 1, inplace = True)


# In[5]:


df.head(5)


# #### **Task 4: Check the number of rows and columns**

# In[6]:


df.shape


# #### **Task 5: Check data types of all columns**

# In[7]:


df.dtypes


# #### **Task 6: Check summary statistics**

# In[8]:


df.describe()


# #### **Task 7: Check summary statistics of all columns, including object dataypes**

# In[9]:


df.describe(include = 'all')


# **Question: Explain the summary statistics for the above data set**

# **Answer:** Summary statistics shows that there is only one column 'price' in numerical. all other columns are object type because we are unable to see the measures of center and spread of the data columns except 'price'. The price column shows that there are null values present in it, and the average(mean) ticket price is 56.7 usd/pound. However, the median is 53, which is less than the mean. This shows that there are outliers at the right side of the distribution which are extending the distribution to the right. Therefore, price is right skewed distribution.

# #### **Task 8: Check null values in dataset**

# In[10]:


df.isnull().sum()


# #### **Task 9: Fill the Null values in the 'price' column.**<br>
# 

# In[11]:


df['price'].fillna(df['price'].median(), inplace = True)


# #### **Task 10: Drop the rows containing Null values in the attributes train_class and fare**

# In[12]:


df.dropna(subset=['train_class','fare'],axis = 0,inplace = True)


# #### **Task 11: Drop 'insert_date'**

# In[13]:


df.drop('insert_date', axis = 1, inplace = True)


# **Check null values again in dataset**

# In[14]:


df.isnull().sum()


# #### **Task 12: Plot number of people boarding from different stations**
# 

# In[15]:


sns.countplot(y = 'origin', data = df); # semicolon will remove the extra line


# **Question: What insights do you get from the above plot?**

# **Answer:** This shows that most of people are booking tickets/coming from Madrid and least people are coming/boarding from Ponferrada. Remaining 3 stations has less variation in their distributions. We can apply passenger safety and comfort measurables in the Madrid station, and form a strategy or a feasibility study to find if the ponferrada station is going to be profitable in near future. Or we can form a strategy to grnerate more passengers from ponferrada station.

# #### **Task 13: Plot number of people for the destination stations**
# 

# In[16]:


sns.countplot(y = 'destination', data = df)


# **Question: What insights do you get from the above graph?**

# **Answer:** <br>
# This shows that most of people are going to from Madrid and least people are going to Ponferrada. Valencia and Barcelona stations has equal number of passenger destination while Sevilla has more than Ponferrada but less than other 3 stations passengers. As from the previous plot and this plot, it has been observed that Madrid is the busiest station among all other stations that are under consideration. Therefore, management can take measures to ensure the safety and security plans to avoid any mismanagement.

# #### **Task 14: Plot different types of train that runs in Spain**
# 

# In[17]:


sns.countplot(y='train_type', data = df)


# **Question: Which train runs the maximum in number as compared to other train types?**

# **Answer:** <br>
# 'AVE' train runs the maximum number as compared to other trains.
# 

# #### **Task 15: Plot number of trains of different class**
# 

# In[18]:


sns.countplot(y = 'train_class',data = df)


# **Question: Which the most common train class for traveling among people in general?**

# **Answer:** <br> Turista is the most common train among people in general.
# 

# In[19]:


df.head()


# #### **Task 16: Plot number of tickets bought from each category**
# 

# In[103]:


sns.countplot(y = 'train_class', data = df)


# #### **Task 17: Plot distribution of the ticket prices**

# In[21]:


sns.histplot(x = 'price', data = df, binwidth = 5, kde = True)


# **Question: What readings can you get from the above plot?**

# **Answer:** <br> it can be seen that the distribution is right skewed. it has high values tickets prices till 200. however most of the prices fall between 25-35 $/pound. which means that people are usually traveling on cheap tickets.  

# ###### **Task 18: Show train_class vs price through boxplot**

# In[22]:


sns.boxplot(x = 'train_class', y = 'price',data = df)


# **Question: What pricing trends can you find out by looking at the plot above?**

# **Answer:** <br> The box plots of the price distribution with train class has some abnormal behavior in it. for example 'Turista' class of train has outliers at 175 however the median price of 'Turista' class is at 50 with minimum ticket price is ~ 20 dollar. which shows that there are some passengers who are paying a lot more than the usual passengers. Similar behavior can be found in 'Turista Plus' but in this class the minimum price of ticket is ~ 25 dollar and it has many high outliers approaching more than 200dollar. it shows that 'Turista Plus has more minimum, 50 percent (at80 dollar) and maximum prices, it is more revenue gererator than 'Turista'. Furthermore, train class 'Preferente' has more  minimum price '~40 dollar' but its maximum is less than 'Turista' and 'Turista Plus'.'Turista con enlace' has no outliers and has very small variation in distribution with right skewed distribution. Finally, 'Cama Turista' has no values in it. it is not generating any business.

# #### **Task 19: Show train_type vs price through boxplot**
# 

# In[23]:


sns.boxplot(y = 'train_type', x = 'price',data = df)


# **Question: Which type of trains cost more as compared to others?**

# **Answer:** 
# 'Ave' Trainhas more variant distribution in it however its minimum price is at second number in lowest among all other trains, with much higher outlier values in it.
# 

# ## Feature Engineering
# 

# In[24]:


df = df.reset_index()


# In[25]:


df.drop(['index'],axis = 1, inplace = True)


# **Finding the travel time between the place of origin and destination**<br>
# We need to find out the travel time for each entry which can be obtained from the 'start_date' and 'end_date' column. Also if you see, these columns are in object type therefore datetimeFormat should be defined to perform the necessary operation of getting the required time.

# **Import datetime library**

# In[26]:


import datetime


# In[27]:


datetimeFormat = '%Y-%m-%d %H:%M:%S'
def fun(a,b):
    diff = datetime.datetime.strptime(b, datetimeFormat) - datetime.datetime.strptime(a, datetimeFormat)
    return(round((diff.seconds/3600.0),2))                  
    


# In[28]:


df['travel_time_in_hrs'] = df.apply(lambda x:fun(x['start_date'],x['end_date']), axis = 1)


# #### **Task 20: Remove redundant features**
# 

# **You need to remove features that are giving the related values as  'travel_time_in_hrs'**<br>
# *Hint: Look for date related columns*

# In[29]:


df.drop(['start_date','end_date'],axis = 1, inplace = True)  


# We now need to find out the pricing from 'MADRID' to other destinations. We also need to find out time which each train requires for travelling. 

# ## **Travelling from MADRID to SEVILLA**

# #### Task 21: Findout people travelling from MADRID to SEVILLA

# In[30]:


df


# In[31]:


df1 = df.loc[df['origin']=='MADRID']
df1 = df.loc[df['destination']=='SEVILLA']


# In[32]:


df1.shape


# #### Task 22: Make a plot for finding out travelling hours for each train type

# In[33]:


sns.barplot(x='train_type',y='travel_time_in_hrs',data =df1)


# #### **Task 23: Show train_type vs price through boxplot**
# 

# In[34]:


sns.boxplot(x='train_type',y='price',data=df1)


# ## **Travelling from MADRID to BARCELONA**
# 

# #### Task 24: Findout people travelling from MADRID to BARCELONA

# In[35]:


df1 = df.loc[df['origin']=='MADRID']
df1 = df.loc[df['destination']=='BARCELONA']


# #### Task 25: Make a plot for finding out travelling hours for each train type

# In[36]:


sns.barplot(x='train_type',y='travel_time_in_hrs',data =df1)


# #### **Task 26: Show train_type vs price through boxplot**

# In[37]:


sns.boxplot(x='train_type',y='price',data=df1)


# ## **Travelling from MADRID to VALENCIA**

# #### Task 27: Findout people travelling from MADRID to VALENCIA

# In[38]:


df1 = df.loc[df['origin']=='MADRID']
df1 = df.loc[df['destination']=='VALENCIA']


# #### Task 28: Make a plot for finding out travelling hours for each train type

# In[39]:


sns.barplot(x='train_type',y='travel_time_in_hrs',data =df1)


# #### **Task 29: Show train_type vs price through boxplot**

# In[40]:


sns.boxplot(x='train_type',y='price',data=df1)


# ## **Travelling from MADRID to PONFERRADA**

# #### Task 30: Findout people travelling from MADRID to PONFERRADA

# In[41]:


df1 = df.loc[df['origin']=='MADRID']
df1 = df.loc[df['destination']=='PONFERRADA']


# #### Task 31: Make a plot for finding out travelling hours for each train type

# In[42]:


sns.barplot(x='train_type',y='travel_time_in_hrs',data =df1)


# #### **Task 32: Show train_type vs price through boxplot**

# In[43]:


sns.boxplot(x='train_type',y='price',data=df1)


# # Applying Linear  Regression

# #### Task 33: Import LabelEncoder library from sklearn 

# In[44]:


from sklearn import preprocessing


# **Data Encoding**

# In[45]:


lab_en = preprocessing.LabelEncoder()


# In[46]:


df


# In[47]:


df.iloc[:,1] = lab_en.fit_transform(df.iloc[:,1])
df.iloc[:,2] = lab_en.fit_transform(df.iloc[:,2])
df.iloc[:,3] = lab_en.fit_transform(df.iloc[:,3])
df.iloc[:,5] = lab_en.fit_transform(df.iloc[:,5])
df.iloc[:,6] = lab_en.fit_transform(df.iloc[:,6])


# In[48]:


df.iloc[:,0] = lab_en.fit_transform(df.iloc[:,0])
df.iloc[:,1] = lab_en.fit_transform(df.iloc[:,1])
df.iloc[:,2] = lab_en.fit_transform(df.iloc[:,2])
df.iloc[:,4] = lab_en.fit_transform(df.iloc[:,4])
df.iloc[:,5] = lab_en.fit_transform(df.iloc[:,5])


# In[49]:


df.head()


# #### Task 34: Separate the dependant and independant variables

# In[50]:


X = df.drop('price',axis = 1)
Y = df[['price']]


# #### Task 35: Import test_train_split from sklearn

# In[51]:


from sklearn.model_selection import train_test_split


# #### Task 36:**Split the data into training and test set**

# In[52]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 42)


# #### Task 37: Import LinearRegression library from sklearn

# In[53]:


from sklearn.linear_model import LinearRegression


# #### Task 38: Make an object of LinearRegression( ) and train it using the training data set

# In[54]:


lr = LinearRegression()


# In[55]:


trained_model = lr.fit(X_train,Y_train)


# #### Task 39: Find out the predictions using test data set.

# In[56]:


lr_predict = trained_model.predict(X_test)


# #### Task 40: Find out the predictions using training data set.

# In[57]:


lr_predict_train = trained_model.predict(X_train)


# #### Task 41: Import r2_score library form sklearn

# In[58]:


from sklearn.metrics import r2_score


# #### Task 42: Find out the R2 Score for test data and print it.

# In[59]:


lr_r2_test= r2_score(Y_test, lr_predict)
lr_r2_test


# #### Task 43: Find out the R2 Score for training data and print it.

# In[60]:


lr_r2_train = r2_score(Y_train, lr_predict_train)
lr_r2_train


# Comaparing training and testing R2 scores

# In[72]:


print('R2 score for Linear Regression Training Data is: ', lr_r2_train)
print('R2 score for Linear Regression Testing Data is: ', lr_r2_test)


# # Applying Polynomial Regression

# #### Task 44: Import PolynomialFeatures from sklearn

# In[62]:


from sklearn.preprocessing import PolynomialFeatures


# #### Task 45: Make and object of default Polynomial Features

# In[63]:


poly_reg = PolynomialFeatures(degree=2)


# #### Task 46: Transform the features to higher degree features.

# In[64]:


X_train_poly = poly_reg.fit_transform(X_train)
X_test_poly = poly_reg.fit_transform(X_test)


# In[65]:


X_train_poly


# #### Task 47: Fit the transformed features to Linear Regression

# In[66]:


poly_model_reg = LinearRegression()
poly_model_reg.fit(X_train_poly, Y_train)


# #### Task 48: Find the predictions on the data set

# In[67]:


y_train_predicted = poly_model_reg.predict(X_train_poly)
y_test_predicted = poly_model_reg.predict(X_test_poly)


# #### Task 49: Evaluate R2 score for training data set

# In[68]:


#evaluating the model on training dataset
r2_train = r2_score(Y_train, y_train_predicted)


# #### Task 50: Evaluate R2 score for test data set

# In[69]:


# evaluating the model on test dataset
r2_test = r2_score(Y_test, y_test_predicted)


# Comaparing training and testing R2 scores

# In[70]:


print ('The r2 score for training set is: ',r2_train)
print ('The r2 score for testing set is: ',r2_test)


# #### Task 51: Select the best model

# **Question: Which model gives the best result for price prediction? Find out the complexity using R2 score and give your answer.**<br>
# *Hint: Use for loop for finding the best degree and model complexity for polynomial regression model*

# In[97]:


r2_train=[]
r2_test=[]
for i in range(1,6):
    poly_reg = PolynomialFeatures(degree=i)
    X_tr_poly,X_tst_poly = poly_reg.fit_transform(X_train),poly_reg.fit_transform(X_test)
    poly = LinearRegression()
    poly.fit(X_tr_poly, Y_train)
   
    y_tr_predicted,y_tst_predict = poly.predict(X_tr_poly),poly.predict(X_tst_poly)


    r2_train.append(r2_score(Y_train, y_tr_predicted))
    r2_test.append(r2_score(Y_test, y_tst_predict))
    
print ('R2 Train', r2_train,'\n\n')
print ('R2 Test', r2_test,'\n')


# #### Plotting the model

# In[98]:


plt.figure(figsize=(18,5))
sns.set_context('poster')
plt.subplot(1,2,1)
sns.lineplot(x=list(range(1,6)), y=r2_train, label='Training');
plt.subplot(1,2,2)
sns.lineplot(x=list(range(1,6)), y=r2_test, label='Testing');


# **Answer** When we increase polynomial degree more that 5, the model becomes overfit: it works well on training data but not works well on testing data. we can observe this behavior from the above graphs.
