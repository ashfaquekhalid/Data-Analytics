import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler=MinMaxScaler()
df=pd.read_csv('train_upd.csv')
df=df.drop(['par_year','par_month'],axis=1)

le= LabelEncoder()
df.iloc[:,36]=le.fit_transform(df.iloc[:,36])
df=pd.get_dummies(df.iloc[:,1:37])

#DF=pd.read_csv('test_upd.csv')

#DF=pd.get_dummies(DF.iloc[:,1:37])
#x_test=scaler.fit_transform(DF.iloc[:,:])

#sns.boxplot(df.iloc[:,10])
#plt.hist(df.iloc[:,6])
#plt.scatter(df.iloc[:,5],df.iloc[:,6])
for i in range(0,34):
  Q1 = df.iloc[:,i].quantile(0)
  Q3 = df.iloc[:,i].quantile(0.988)
  IQR = Q3-Q1
  df.iloc[:,i] = df.iloc[:,i][~((df.iloc[:,i] < (Q1 - 1.5*IQR)) | (df.iloc[:,i] > (Q3 + 1.5*IQR)))]
#df=df.drop(['beam_direction','cell_range','tilt'],axis=1)  
df1=df.dropna()
#df1=df
 #df1['total_bytes']=0
#for j in range(5,31):
    #df1['total_bytes']=df1['total_bytes']+df1.iloc[:,j]
#for k in range(5,32):
     #df1=df1.drop(df1.columns[5],axis=1)
df2=df1.drop(['Congestion_Type'],axis=1)
#plt.hist(df1.iloc[:,6])
#sns.boxplot(df1.iloc[:,10])


x=scaler.fit_transform(df2.iloc[:,:])
y=df1['Congestion_Type'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

plt.scatter(df1.iloc[:,31],df1.iloc[:,34])
#plt.hist(df.iloc[:,6])
#plt.hist(df1.iloc[:,6])
from sklearn import svm

clf = svm.SVC(kernel='rbf', gamma=0.3)
#from sklearn.feature_selection import RFE
#selector = RFE(clf,30, step=1)
#selector = selector.fit(x, y)
#selector.support_
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test,y_pred))

from sklearn.metrics import matthews_corrcoef
print (matthews_corrcoef(y_test,y_pred))