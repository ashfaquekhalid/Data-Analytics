%matplotlib inline
#Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Importing the dataset
dataset = pd.read_csv('train_upd.csv')
test = pd.read_csv('test_upd.csv')
cell_name = test['cell_name']
dataset.head()
#Checking Null values
print(dataset.isnull().sum())
print(test.isnull().sum())
#separting the variables
X = dataset.iloc[:, 1:-1]
test = test.iloc[:, 1:]
y = dataset.iloc[:, 38:39]

# Encoding the "ran_vendor"
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X.iloc[:, 36] = labelencoder_X.fit_transform(X.iloc[:, 36])
test.iloc[:,36] = labelencoder_X.transform(test.iloc[:,36])
# Encoding the "Congestion_Type"
labelencoder_y = LabelEncoder()
y.iloc[:, 0] = labelencoder_y.fit_transform(y.iloc[:, 0])

#Concatenating
df = pd.concat([X,y], axis=1)

df.head()
# Feature Scaling
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
df.iloc[:, 6:36] = min_max_scaler.fit_transform(df.iloc[:, 6:36])
test.iloc[:, 6:36] = min_max_scaler.transform(test.iloc[:, 6:36])
df.head()
df.describe()
#Checking count for target variable
df.Congestion_Type.value_counts()
import seaborn as sns
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),cmap='Blues',annot=False) 
l = df.columns.values
number_of_columns=27
number_of_rows = len(l)-1/number_of_columns
plt.figure(figsize=(5*number_of_columns,5*number_of_rows))
for i in range(6,33):
    plt.subplot(number_of_rows + 1,number_of_columns,i+1)
    sns.distplot(df[l[i]],kde=True) 
    plt.figure(figsize=(5*number_of_columns,5*number_of_rows))
for j in range(6,33):
    plt.subplot(number_of_rows + 1,number_of_columns,j+1)
    sns.boxplot(y=df[l[j]], x=df[l[37]])

df['par_day'].replace(to_replace=[1,8,15,22,29,2,9,16,23,30],value=1,inplace=True)    # labeling 1 for weekend
df['par_day'].replace(to_replace=[3,4,5,6,7,10,11,12,13,14,17,18,19,20,21,24,25,26,27,28,31],value=0,inplace=True)   #labeling 0 for weekdays


test['par_day'].replace(to_replace=[1,8,15,22,29,2,9,16,23,30],value=1,inplace=True)    # labeling 1 for weekend
test['par_day'].replace(to_replace=[3,4,5,6,7,10,11,12,13,14,17,18,19,20,21,24,25,26,27,28,31],value=0,inplace=True)   #labeling 0 for weekdays

df.drop(['par_day'],inplace = True,axis=1)
test.drop(['par_day'],inplace = True, axis=1)
df['morning'] = [1 if(x>=2 and x<8) else 0 for x in df['par_hour']]
df['day'] = [1 if(x>=8 and x<18) else 0 for x in df['par_hour']]
df['evening'] = [1 if(x>=18 and x<22) else 0 for x in df['par_hour']]
df['night'] = [1 if(x>=22 or x<2) else 0 for x in df['par_hour']]

test['morning'] = [1 if(x>=2 and x<8) else 0 for x in test['par_hour']]
test['day'] = [1 if(x>=8 and x<18) else 0 for x in test['par_hour']]
test['evening'] = [1 if(x>=18 and x<22) else 0 for x in test['par_hour']]
test['night'] = [1 if(x>=22 or x<2) else 0 for x in test['par_hour']]

df.drop(['par_hour'],inplace = True,axis=1)
test.drop(['par_hour'],inplace = True, axis=1)
df['TCP'] = df['web_browsing_total_bytes'] + df['social_ntwrking_bytes'] + df['health_total_bytes'] + df['communication_total_bytes'] + df['file_sharing_total_bytes'] + df['remote_access_total_bytes'] + df['location_services_total_bytes'] + df['presence_total_bytes'] + df['advertisement_total_bytes'] + df['voip_total_bytes'] + df['speedtest_total_bytes'] + df['email_total_bytes'] + df['weather_total_bytes'] + df['mms_total_bytes'] + df['others_total_bytes']
df['UDP'] = df['video_total_bytes'] + df['cloud_computing_total_bytes'] + df['web_security_total_bytes'] + df['gaming_total_bytes'] + df['photo_sharing_total_bytes'] + df['software_dwnld_total_bytes'] + df['marketplace_total_bytes'] + df['storage_services_total_bytes'] + df['audio_total_bytes'] + df['system_total_bytes'] + df['media_total_bytes']
df['ratio']=df['UDP']/df['TCP']

test['TCP'] = test['web_browsing_total_bytes'] + test['social_ntwrking_bytes'] + test['health_total_bytes'] + test['communication_total_bytes'] + test['file_sharing_total_bytes'] + test['remote_access_total_bytes'] + test['location_services_total_bytes'] + test['presence_total_bytes'] + test['advertisement_total_bytes'] + test['voip_total_bytes'] + test['speedtest_total_bytes'] + test['email_total_bytes'] + test['weather_total_bytes'] + test['mms_total_bytes'] + test['others_total_bytes']
test['UDP'] = test['video_total_bytes'] + test['cloud_computing_total_bytes'] + test['web_security_total_bytes'] + test['gaming_total_bytes'] + test['photo_sharing_total_bytes'] + test['software_dwnld_total_bytes'] + test['marketplace_total_bytes'] + test['storage_services_total_bytes'] + test['audio_total_bytes'] + test['system_total_bytes'] + test['media_total_bytes']
test['ratio']= test['UDP']/test['TCP']

df['temp1']=df['web_security_total_bytes']+df['gaming_total_bytes']+df['storage_services_total_bytes']+ df['system_total_bytes']+df ['media_total_bytes']
df['temp2']= df['health_total_bytes']+df['location_services_total_bytes']+df['advertisement_total_bytes']+df['voip_total_bytes']+df ['speedtest_total_bytes']+df['email_total_bytes']+df['weather_total_bytes']+df['mms_total_bytes']+df['others_total_bytes']

test['temp1']=test['web_security_total_bytes']+test['gaming_total_bytes']+ test['storage_services_total_bytes']+ test['system_total_bytes']+ test['media_total_bytes']
test['temp2']=test['health_total_bytes']+ test['location_services_total_bytes']+ test['advertisement_total_bytes']+ test['voip_total_bytes']+ test['speedtest_total_bytes']+ test['email_total_bytes']+ test['weather_total_bytes']+ test['mms_total_bytes']+test['others_total_bytes']


df['temp1']=df['temp1']/5
df['temp2']=df['temp2']/9

test['temp1']=test['temp1']/5
test['temp2']=test['temp2']/9

df['temp3']=df['web_browsing_total_bytes']+df['social_ntwrking_bytes']

test['temp3']= test['web_browsing_total_bytes']+test['social_ntwrking_bytes']


df['temp3']=df['temp3']/2

test['temp3']=test['temp3']/2

df['temp4']= df['presence_total_bytes']+df['communication_total_bytes']

test['temp4']= test['presence_total_bytes']+test['communication_total_bytes']


df['temp4']=df['temp4']/2

test['temp4']=test['temp4']/2

df['temp5']=df['photo_sharing_total_bytes']+df['software_dwnld_total_bytes']

test['temp5']=test['photo_sharing_total_bytes']+test['software_dwnld_total_bytes']


df['temp5']=df['temp5']/2

test['temp5']=test['temp5']/2

df['temp6']=df['file_sharing_total_bytes']+df['remote_access_total_bytes']

test['temp6']=test['file_sharing_total_bytes']+test['remote_access_total_bytes']


df['temp6']=df['temp6']/2

test['temp6']=test['temp6']/2

df['temp7']=df['marketplace_total_bytes'] +df['audio_total_bytes']

test['temp7']=test['marketplace_total_bytes'] +test['audio_total_bytes']


df['temp7']=df['temp7']/2

test['temp7']=test['temp7']/2

df['total_bytes']=0
test['total_bytes']=0
for x in df.columns:
    if 'bytes' in x:
        df[x]=np.log(df[x]+1)
        test[x]=np.log(test[x]+1)
        df['total_bytes']=df['total_bytes']+df[x]
        test['total_bytes']=test['total_bytes']+test[x]
        
l = df.columns.values
number_of_columns=11
number_of_rows = len(l)-1/number_of_columns
plt.figure(figsize=(5*number_of_columns,5*number_of_rows))
for i in range(40,51):
    plt.subplot(number_of_rows + 1,number_of_columns,i+1)
    sns.distplot(df[l[i]].astype('float64'),kde=True)

df = df.drop(['par_year','par_month','Congestion_Type'], axis=1)
test.drop(['par_year','par_month'],axis = 1,inplace = True)
df.head()

#selecting features
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, 4)
fit = rfe.fit(X, y)
(print("Num Features: %d" % fit.n_features_,))
(print("Selected Features: %s" % fit.support_,))
(print("Feature Ranking: %s" % fit.ranking_,))

#printing features selected by RFE
for i in range(len(fit.support_)):
    if fit.support_[i]:
        print(df.columns[i])

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, stratify=y, test_size=0.25)
clf=RandomForestClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
features =df.columns.values
importance = clf.feature_importances_ / np.max(clf.feature_importances_)
idx = np.argsort(importance)
n = 10
pos = np.arange(n) + 0.5
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(pos, importance[idx[-n:]], align='center')
plt.yticks(pos, features[idx[-n:]])
ax.set_xlabel('Relative Importance')
ax.set_ylabel('Feature Name')

plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef  

X_train, X_test, y_train,y_test = train_test_split(df,y,test_size = 0.3)
clf = LogisticRegression( multi_class='multinomial', solver='lbfgs')
clf.fit(X_train, y_train)
y_pred_lr = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred_lr)
accuracy=accuracy_score(y_test,y_pred_lr)
print(matthews_corrcoef(y_test, y_pred_lr))

# Set the parameters by cross-validation
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

# Tree
param_grid = {"max_depth": np.linspace(10, 15, 6).astype(int),
              "min_samples_split": np.linspace(2, 5, 4).astype(int)
              }
clf_dt = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=cv)
clf_dt.fit(X_train, y_train)

y_pred_dt = clf_dt.predict(X_test)

cm = confusion_matrix(y_test, y_pred_dt)
accuracy=accuracy_score(y_test,y_pred_dt)
print(matthews_corrcoef(y_test, y_pred_dt))

clf2=RandomForestClassifier()
clf2.fit(X_train,y_train)
y_pred_rf=clf2.predict(X_test)

cm = confusion_matrix(y_test, y_pred_rf)
accuracy=accuracy_score(y_test,y_pred_rf)
print(matthews_corrcoef(y_test, y_pred_rf))

xgb = XGBClassifier(
 learning_rate =0.05,
 n_estimators=500,
 max_depth=5,
 min_child_weight=2,
 gamma=0.1,
 subsample=0.6,
 colsample_bytree=0.7,
 reg_alpha=0.1,
 objective= 'multi:softmax',
 scale_pos_weight=1,
 random_state=7,
 seed=27)
xgb.fit(X_train,y_train)
y_pred_xgb=xgb.predict(X_test)

cm = confusion_matrix(y_test, y_pred_xgb)
accuracy=accuracy_score(y_test,y_pred_xgb)
print(matthews_corrcoef(y_test, y_pred_xgb))

# Set the parameters by cross-validation
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

# MLP
param_grid = {"hidden_layer_sizes": [(50,), (50, 50)],
              "alpha": np.logspace(-2, 2, 6)
              }
clf_mlp = GridSearchCV(MLPClassifier('lbfgs', max_iter=600), param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)
clf_mlp.fit(X_train, y_train)
y_pred_mlp = clf_mlp.predict(X_test)


cm = confusion_matrix(y_test, y_pred_mlp)
accuracy=accuracy_score(y_test,y_pred_mlp)
print(matthews_corrcoef(y_test, y_pred_mlp))

# from sklearn.grid_search import GridSearchCV   #Perforing grid search
param_test1 = {
 'learning_rate':[0.01,0.025,0.05],
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1,n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_train,y_train)

gsearch1.best_params_, gsearch1.best_score_

#Perforing grid search
param_test2 = {
 'reg_alpha':np.linspace(0.1,0.5,5),
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test2,n_jobs=4,iid=False, cv=5)
gsearch2.fit(X_train,y_train)

gsearch2.best_params_, gsearch2.best_score_

param_test3 = {
 'reg_lambda':np.linspace(0,0.6,7)
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test3,n_jobs=4,iid=False, cv=5)
gsearch3.fit(X_train,y_train)

gsearch3.best_params_, gsearch3.best_score_

xgb = XGBClassifier(
 learning_rate =0.05,
 n_estimators=500,
 max_depth=5,
 min_child_weight=2,
 gamma=0.1,
 subsample=0.6,
 colsample_bytree=0.7,
 reg_alpha=0.4,
 reg_lambda=0.4,
 objective= 'multi:softmax',
 scale_pos_weight=1,
 random_state=7,
 seed=27)
xgb.fit(X_train,y_train)
y_pred_xgb=xgb.predict(X_test)

cm = confusion_matrix(y_test, y_pred_xgb)
accuracy=accuracy_score(y_test,y_pred_xgb)
print(matthews_corrcoef(y_test, y_pred_xgb)

y_pred_xg=xgb.predict(X_train)
cm = confusion_matrix(y_train, y_pred_xg)
accuracy=accuracy_score(y_train,y_pred_xg)
print(matthews_corrcoef(y_train, y_pred_xg))

y_pred_test = xgb.predict(test)
y_pred_test = labelencoder_y.inverse_transform(y_pred_test)

df_final = pd.DataFrame({'cell_name':cell_name,
                        'Congestion_Type': y_pred_test})
df_final.to_csv('Submission.csv',header = True, index = False)

      
    