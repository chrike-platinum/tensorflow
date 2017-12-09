__author__ = 'christiaan'

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn import preprocessing

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

X, X_test, Y, Y_test,t = splitTestTrain(titanicDF,test_size=0.1,label='Survived')
Y = np.array([Y, -(Y-1)]).T
Y_test = np.array([Y_test, -(Y_test-1)]).T





def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
def model(X, w_h,w_h2,w_h3,w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    h2= tf.nn.sigmoid(tf.matmul(h, w_h2))
    h3= tf.nn.sigmoid(tf.matmul(h2, w_h3))
    return tf.matmul(h3, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


trX, trY, teX, teY = X, Y, X_test, Y_test

X = tf.placeholder("float", [None, 9])
Y = tf.placeholder("float", [None,2])
w_h = init_weights([9, 20]) # create symbolic variables
w_h2 = init_weights([20, 80])
w_h3 = init_weights([80, 20])
w_o = init_weights([20, 2])
py_x = model(X, w_h,w_h2,w_h3, w_o)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
train_op = tf.train.AdamOptimizer(0.0001).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x,1)
# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        a,c= sess.run([cost,train_op], feed_dict={X: trX, Y: trY})
        #print(i,a)
        #print(i, sess.run([Y,predict_op], feed_dict={X: teX,Y:teY}))
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX, Y: teY})))