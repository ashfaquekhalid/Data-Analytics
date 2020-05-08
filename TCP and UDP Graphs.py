# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train = pd.read_csv('train_upd.csv')
test = pd.read_csv('test_upd.csv')
true = pd.read_csv('Sample_submission.csv')

X_train = train.iloc[:, 1:-1].values
y_train = train.iloc[:, 38].values
X_test = test.iloc[:, 1:].values

y_true = true.iloc[:,1].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder #OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 36] = labelencoder_X.fit_transform(X_train[:, 36])
'''onehotencoder = OneHotEncoder(categorical_features = [36])
X_train = onehotencoder.fit_transform(X_train).toarray()'''
X_test[:, 36] = labelencoder_X.fit_transform(X_test[:, 36])
'''onehotencoder = OneHotEncoder(categorical_features = [36])
X_test = onehotencoder.fit_transform(X_test).toarray()'''

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)
y_true = labelencoder_y.fit_transform(y_true)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

#Inserting new array
'''train.drop(['TCP'], axis = 1)
train.insert(39,'TCP', 0)
train.dtype()

train[:,39] = train[:,8] '''
''' TCP = np.array(X_train[:,6] , dtype = 'int16')
TCP = np.add(( X_train[:,7] , X_train[:,9] , X_train[:,13] , X_train[:,14] , X_train[:, 15] + 
        X_train[:,16] , X_train[:,22] , X_train[:,23] , X_train[:,24] + X_train[:,26] + 
        X_train[:,27] , X_train[:,28] , X_train[:,29] , X_train[:,31] , X_train[:,32]))'''


TCP = np.array([X_train[:,7] + X_train[:,9] + X_train[:,13] + X_train[:,14] + X_train[:, 15] + 
        X_train[:,16] + X_train[:,22] + X_train[:,23] + X_train[:,24] + X_train[:,26] + 
        X_train[:,27] + X_train[:,28] + X_train[:,29] + X_train[:,31] + X_train[:,32] ], dtype = 'int16')

train['TCP'] = train['web_browsing_total_bytes'] + train['social_ntwrking_bytes'] + train['health_total_bytes'] + train['communication_total_bytes'] + train['file_sharing_total_bytes'] + train['remote_access_total_bytes'] + train['location_services_total_bytes'] + train['presence_total_bytes'] + train['advertisement_total_bytes'] + train['voip_total_bytes'] + train['speedtest_total_bytes'] + train['email_total_bytes'] + train['weather_total_bytes'] + train['mms_total_bytes'] + train['others_total_bytes']
train['UDP'] = train['video_total_bytes'] + train['cloud_computing_total_bytes'] + train['web_security_total_bytes'] + train['gaming_total_bytes'] + train['photo_sharing_total_bytes'] + train['software_dwnld_total_bytes'] + train['marketplace_total_bytes'] + train['storage_services_total_bytes'] + train['audio_total_bytes'] + train['system_total_bytes'] + train['media_total_bytes']

#Converting into training set
X_TCP = train.iloc[:,39:40].values
X_UDP = train.iloc[:,40:41].values


#Creating the boxplot
plt.boxplot(X_TCP)
plt.boxplot(X_UDP)

#Creating the histogram
plt.hist(X_TCP)
plt.hist(X_UDP)

#Bar Graph
fig, ax = plt.subplots()
n_groups = 4






