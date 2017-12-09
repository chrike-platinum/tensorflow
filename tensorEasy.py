

__author__ = 'christiaan'


import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn import preprocessing
import tensorflow as tf
import tempfile

titanicDF = pd.read_csv('/Users/christiaan/Downloads/ML/Exercice/titanic.csv',na_values='NaN')


def replaceAgeGroups(x):
    if (x <16): return 0
    elif (16 <=x<32): return 1
    elif (32 <=x<48): return 2
    elif (48<=x<64): return 3
    else:
        return 4

def replaceFareGroups(x):
    if (x <7.91): return 0
    elif (7.91 <=x<14.454): return 1
    elif (14.454 <=x<31): return 2
    else:
        return 3

#impute Age by median
titanicDF['Age'].fillna(titanicDF['Age'].median(),inplace=True)
#titanicDF['AgeGroup'] = titanicDF[['Age']].applymap(replaceAgeGroups)


#titanicDF['FareGroup'] = titanicDF[['Fare']].applymap(replaceFareGroups)
#titanicDF=titanicDF.drop(['Fare'], axis=1)



#titanicDF['groupSize'] = titanicDF['SibSp'] + titanicDF['Parch'] + 1

#titanicDF['Alone']= np.where((titanicDF['groupSize']==1),1,0)



#titanicDF=titanicDF.drop(['groupSize'], axis=1)
#titanicDF=titanicDF.drop(['SibSp'], axis=1)
titanicDF=titanicDF.drop(['Parch'], axis=1)
min_max_scaler = preprocessing.MinMaxScaler()
titanicDF['Fare'] = min_max_scaler.fit_transform(titanicDF['Fare'])
titanicDF['Age'] = min_max_scaler.fit_transform(titanicDF['Age'])

#drop cabin bc too much NaNs
titanicDF=titanicDF.drop(['Cabin'], axis=1)

#contain only numeric part of ticket
titanicDF=titanicDF[titanicDF[['Ticket']].apply(lambda x: x[0].isdigit(), axis=1)]



#print(titanicDF[titanicDF['Sex']=='female' and titanicDF['Age']==20])

#create dummies for binary sex
sexlabels = pd.get_dummies(titanicDF['Sex'])
titanicDF=titanicDF.join(sexlabels).drop(['Sex','male'],axis=1)



#drop rest of missing values
titanicDF.dropna(inplace=True)

#remove names / look just at ID
titanicDF.dropna()


#Map embarked to classes
#mapping = {'S': 0, 'Q': 1,'C':2}
#titanicDF.replace({'Embarked': mapping},inplace=True)

#get Dummies for embarked
sexlabels = pd.get_dummies(titanicDF['Embarked'],drop_first=True)
titanicDF=titanicDF.join(sexlabels).drop(['Embarked'],axis=1)

#drop Names
titanicDF.drop(['Name'],axis=1,inplace=True)

#
#titanicDF.drop(['PassengerId'],axis=1,inplace=True)
titanicDF.drop(['PassengerId'], axis=1,inplace=True)

#reset index df
titanicDF=titanicDF.apply(pd.to_numeric)
titanicDF.reset_index(inplace=True)
#titanicDF = normalize(titanicDF, axis=1, norm='max')


#titanicDF=titanicDF[['Alone','AgeGroup','FareGroup','Survived']]
print(titanicDF)


def splitTestTrain(df,test_size=0.1,label='Survived'):
    labels = df[label].values
    featuresL = df.drop(['Survived'],axis=1)
    features = df.drop(['Survived'],axis=1).values
    featuresNames = list(featuresL)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)
    return X_train, X_test, y_train, y_test,featuresNames

X_train, X_test, y_train, y_test,t = splitTestTrain(titanicDF,test_size=0.1,label='Survived')
#Y = np.array([Y, -(Y-1)]).T
#Y_test = np.array([Y_test, -(Y_test-1)]).T

#y_train.reshape(-1, y_train.shape[0])
#y_test.reshape(-1, y_test.shape[0])



features = []
for c in Columns:
        features.append(tf.contrib.layers.real_valued_column(str(c)))
model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=features,
        model_dir=model_dir)
m.fit(input_fn=lambda: train_input_fn(train_data), steps=200)
results = m.evaluate(input_fn=lambda: eval_input_fn(test_data), steps=1)