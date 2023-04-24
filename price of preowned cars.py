# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
import numpy as py 
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
cars_data=pd.read_csv("cars_sampled.csv")
cars=cars_data.copy()
cars.info()
cars.describe()
pd.set_option('display.float_formate',lamda x: %.3f, %x)
pd.set_option('display.max_column',500)
cars.describe()
#Dropping unwanted column
col=['name','datacrawled','postalcode','lastseen']
cars=cars.drop(columns=col;axis=1)
#removing duplicate records
cars.drop_duplicates(keep='first',inplace=True)
#no. of missing vaues in each column
cars.isnull().sum()
#variable year of registrations

yearwise_count=cars['yearsOfRegistration'].value_count().sort_index()
sum(cars['yearsOfRegistration']>2018)
sum(cars['yearsOfRegistration']<1950)
sns.regplot(x='yearsOfRegistration',y='price',scatter=True,fit_reg=False,data=cars)
#variable price
price_count=cars['price'].value_counts().sort_index()
sns.distplot(cars['price'])
cars['price'].describe()
sns.boxplot(y=cars['price'])
sum(cars['price']>150000)
sum(cars['price']<100)
#variable powerps
power_count=cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxpot(y=cars['powerPS'])
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)
sum(cars['powerPS']>500)
sum(cars['powerPS']<10)
#working range of data
cars=cars[(cars.yearOfRegistration<=2018)&(cars.yearOfRegistration>=1950)&(cars.price>=100)&(cars.price<=150000)&(cars.powerPS>=10)&(cars.powerPS<=500)]
cars['month of Registration']/12
#creating new varible age by adding year of reg. and month of reg.
cars['Age']=(2018-cars['yearOfRegistration])+cars['MonthOfReg']
cars['Age']=round(cars['Age'],2)
cars['Age'].describe()
#Dropping year of reg and month of reg.
cars=cars.drop(columns=['yearOfReg','MonthOfReg'],axis=1)
#Visualising perameter
#Age
Sns.displot(cars['Age'])
Sns.boxplot(cars['price'])
#price
sns.displot(cars['price'])
sns.boxplot(y=cars['price'])
#powerPs
sns.distplot(cars['powerPS']) 
sns.boxplot(y=cars['price']) 
#visualising parameters after narrowing working range
#Age vs Price
sns.regplot(x='Age',y='price',scatter=True,fit_reg=False,data=cars)
#powerPS vs Price
sns.regplot(x='PowerPS',y='price',scatter=True,fit_reg=False,data=cars)
#variable saller
cars['seller'].value_counts()
pd.crosstab(cars['seller'].columns='count',normalize=True)
sns.count(x='seller',data=cars)
#variable offer type
cars['offerType'].value_count()
sns.countplot(x='offerType',data=cars) 
#variable obtest
cars['obtest'].value_counts()
pd.crosstab(cars['abtest'],column='count',normalize=True)
sns.countlpot(x='abtest',data=cars)
#equally distributed    
sns.boxplot(x='obtest',y='price',data=cars)
#variable vehicle type
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],column='count',Normalize=True)
sns.countplot(x=vehicleType,data=cars)
sns.boxplot(x='vehicleType',y='price',data=cars)
#veriable gearbox
cars['gearbox'].value_count()
pd.crosstab(cars['gearbox'],columns='count',normalize=True)
sns.countplot(x='gearbox',y='price',data=cars)
#variable model
cars['Model'].value_count()
pd.crosstab(cars['model'],columns='count',normalize=True)
sns.countplot(x='model',data=cars)
sns.boxplot(x='model',y='price',data=cars)
#variable kilometer
cars['kilometer'].value_count().sort_index()
pd.crosstab(cars['kilometer'],columns='count',normalize=True)
sns.boxplot(x='kilometer',y='price',data=cars)
cars['kilometer'].deacribe()
sns.distplot(cars['kilometer'],bins=8,kde=False)
sns.regplot(x='kilometer',y='price',scatter=True,fit_reg=False,data=cars)
#variable fuel type
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],column='count',normalize=True)
sns.countplot(x='fuelType',data=cars)
sns.boxplot(x='fuelType',y='price',data=cars)
#variable brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='count',normalize=True)
sns.countplot(x='brand',data=cars)
sns.boxplot(x='brand',y='price',data=cars)
#variable not repair damages
cars['notRepairDamage'].value_counts()
pd.crosstab(cars['notRepairDamage'],columns='count',normalize=True)
sns.countplot(x='notRepairDamage',data=cars)
sns.boxplot(x='notRepairDamage',y='price',data=cars)
#Removing Insignificant variables
col=['seller',offerType='obtest']
cars=cars.drop(columns=col,axis=1)
cars_copy=cars.copy()
#correlation
cars_select1=cars_dtype(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)
cars_select1.corr().loc[:'price'].obs().sort_values(ascending=False)[1:]
#building a linear regression and random forest on two types ofdata
#Omitting Missing Values
cars_omit=cars.dropna(axis=0)
cars_omit=pd.get_dummies(cars_omit,drop_first=True)
#Importing neccessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import linearRegression
from sklearn.ensemble import RandoForestRegression
from sklearn.matrics import mean_squared_error
#separating input and output feature
x1=cars_omit.drop(['price'],axis='column',inplace=False)
y1=cars_omit['price']
#plotting the variable price
prices=pd.DataFrame({'1.Before':y1,'2.After':np.log(y1)})
prices.hist()
#transforming the price as a logrithming value
y1=np.log(y1)

#splitting the into train and test
x_train,y_train,x_test,y_test=train_test_split(x1,y1,test_size=0.3,random_state=3)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape) 
#Baseline for omitted data
#making baseline model byusing test data mean value for setting beanchmark and to  compare with our regression model
#finding the mean for test data value
base_pred=np.mean(y_test)
print(base_pred)
#repating the same value till leangth of test data
base_pred=np.repeat(base_pred.len(y_test))
#finding RMSE
base_root_mean_square_error=np.sqrt(maen_squared_error(y_test,base_pred))
print(base_root_mean_square_error)
#linear regression with omitted data
lgr=LinearRegression(fit_intercept=True)
#model
model_lin1=lgr.fit(x_train,y_train)
#predicting model on test data
cars_prediction_lin1=lgr.predict(x_text)
#computing MSE and RMSE
lin_mse1=mean_squared_error(y_test,car_prediction_lin1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)
#R squared value
r2_lin_test1=model_lin1.score(x_test,y_test)
r2_lin_train1=model_lin1.score(x_train,y_train)
print(r2_lin_test1,r2_lin_train1)
#Regression diagnotic_residual plot analysis
residuals1=y_test_cars_prediction_lin1
sns.regplot(x=cars_prediction_lin1,y=rrsiduals1,scatter=True,fit(reg=False))
residuals1.describe()
#Random Forest with Omitted data
#Model perameters
rf=RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_sample_splits=10,min-sample_leaf=4,random_state=1)
#model
model_rf1=rf.fit(x_train,y_train)
#predicting model on test set
cars_prediction_rf1=rf_predict(x_test)
#computing MSE and RMSE
rf_mse1=mean_squared_error(y_test,cars_prediction_rf1)
rf_rmse1=np.sqrt(rf_mse1)
print(rf_rmse1)
#R squared values
r2_rf_test1=model_rf1.score(x_test,y_test)
r2_rf_train1=model_rf1.score(x_train,y_train)
print(r2_rf_test1,r2_rf_train1)
#Model building with Imputed data
cars_imputed=cars_apply(lamda x:x filln{x.median())}\if x.dtype=='float' else \x.fillna(x.value_counts().index[0]))
cars_imputed.isnull().sum()
#converting categorical value to dummy value
cars_imputed=pd.get_dummies(cars_imputed,drop_first=True)
#Model building with imputed data
x2=cars_imputes.drop(['price'],axis='columns',inplace=False)
y2=cars_imputed['price']
#ploting the variable price
prices=pd.DataFrame({'1.Before':y2,'2.After':np.log(y2)})
prices.hist()
#transforming prices as logarthmic value
y2=np.log(y2)
#splitting data into test and train
x_train1,x_test1,y_train1,y_test1=train_test_split(x2,y2,test_size=0.3,random_state=3)
print(x_train1.shape,x_test1.shape,y_train1,y_test.shape)
#Baseline Model for imputed data
base_pred=np.mean(y_test1)
print(base_pred)
#repeating the same value till leangth of test data
base_pred=np.repeat(base_pred.len(y_test1))
#finding RMSE
base_root_mean_square_error_imputed=np.sqrt(mean_squared_error(y_test1,base_pred))
print(base_root_mean_square_error_imputed)
#linear regression with imputed data
#setting intercept as true
lgr2=LinearRegression(fit_intercept=True)
#model
model_lin2=lgr2.fit(x_train1,y_train1)
#predicting model on test data
cars_prediction_lin2=lgr2.predict(x_text1)
#computing MSE and RMSE
lin_mse2=mean_squared_error(y_tes1t,car_prediction_lin2)
lin_rmse2=np.sqrt(lin_mse2)
print(lin_rmse2)
#R squared value
r2_lin_test2=model_lin2.score(x_test1,y_test1)
r2_lin_train2=model_lin2.score(x_train1,y_train1)
print(r2_lin_test2,r2_lin_train2)
#Random Forest with imputed data
#Model perameters
rf2=RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_sample_splits=10,min-sample_leaf=4,random_state=1)
#model
model_rf2=rf2.fit(x_train1,y_train1)
#predicting model on test set
cars_prediction_rf2=rf2.predict(x_test1)
#computing MSE and RMSE
rf_mse2=mean_squared_error(y_test1,cars_prediction_rf2)
rf_rmse2=np.sqrt(rf_mse2)
print(rf_rmse2)
#R squared values
r2_rf_test2=model_rf2.score(x_test1,y_test1)
r2_rf_train2=model_rf2.score(x_train1,y_train1)
print(r2_rf_test2,r2_rf_train2)
