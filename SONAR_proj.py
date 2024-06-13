#SONAR ROCK v/s Mine Prediction

#lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

path=r'C:\Users\SOHAIB\Downloads\Copy of sonar data.csv'
df=pd.read_csv(path,header=None)
print(df.head())
print(df.shape)
print(df.describe())

print(df.iloc[:,-1:].value_counts())

print(df.groupby(60).mean())
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=1)

print('shape of ',X_train.shape,y_train.shape)
print('shape of ',X_test.shape,y_test.shape)


#models applied

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import Lasso

lr=LogisticRegression()
lr.fit(X_train,y_train)
lr_predict=lr.predict(X_test)
print(accuracy_score(y_test,lr_predict))
print(y_train)
svc=SVC()
svc.fit(X_train,y_train)
svc_predict_=svc.predict(X_test)
print(svc.score(X_test,svc_predict_))

#now making a predictive system

input_data=(0.0453,0.0523,0.0843,0.0689,0.1183,0.2583,0.2156,0.3481,0.3337,0.2872,0.4918,0.6552,0.6919,0.7797,0.7464,0.9444,1.0000,0.8874,0.8024,0.7818,0.5212,0.4052,0.3957,0.3914,0.3250,0.3200,0.3271,0.2767,0.4423,0.2028,0.3788,0.2947,0.1984,0.2341,0.1306,0.4182,0.3835,0.1057,0.1840,0.1970,0.1674,0.0583,0.1401,0.1628,0.0621,0.0203,0.0530,0.0742,0.0409,0.0061,0.0125,0.0084,0.0089,0.0048,0.0094,0.0191,0.0140,0.0049,0.0052,0.0044)
input_data2=(0.0286,0.0453,0.0277,0.0174,0.0384,0.0990,0.1201,0.1833,0.2105,0.3039,0.2988,0.4250,0.6343,0.8198,1.0000,0.9988,0.9508,0.9025,0.7234,0.5122,0.2074,0.3985,0.5890,0.2872,0.2043,0.5782,0.5389,0.3750,0.3411,0.5067,0.5580,0.4778,0.3299,0.2198,0.1407,0.2856,0.3807,0.4158,0.4054,0.3296,0.2707,0.2650,0.0723,0.1238,0.1192,0.1089,0.0623,0.0494,0.0264,0.0081,0.0104,0.0045,0.0014,0.0038,0.0013,0.0089,0.0057,0.0027,0.0051,0.0062)

input_data_as_numpy_arry=np.asarray(input_data)
input_data2_as_numpy_array=np.asarray(input_data2)
input_data_reshape=input_data_as_numpy_arry.reshape(1,-1)
input_data2_reshape=input_data2_as_numpy_array.reshape(1,-1)

prediction=lr.predict(input_data2_reshape)

print(prediction)


prediction_SVC=svc.predict(input_data2_reshape)

print(prediction_SVC)



