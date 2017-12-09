__author__ = 'christiaan'

import pandas as pd
#import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import KFold
from sklearn import metrics, cross_validation
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np
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
min_max_scaler = preprocessing.MinMaxScaler()
titanicDF['Age'] = min_max_scaler.fit_transform(titanicDF['Age'])

#
#titanicDF.drop(['PassengerId'],axis=1,inplace=True)
titanicDF.drop(['PassengerId'], axis=1,inplace=True)
titanicDF['Fare'] = min_max_scaler.fit_transform(titanicDF['Fare'])

#reset index df
titanicDF.reset_index(inplace=True)

print(titanicDF)
print(titanicDF.shape)




def dropFeature(df,feature):
    return df.drop([feature],inplace=True)

def splitTestTrain(df,test_size=0.1,label='Survived'):
    labels = df[label].values
    featuresL = df.drop(['Survived'],axis=1)
    features = df.drop(['Survived'],axis=1).values
    featuresNames = list(featuresL)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)
    return X_train, X_test, y_train, y_test,featuresNames




__author__ = 'christiaan'
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


X_train, X_test, y_train, y_test,t = splitTestTrain(titanicDF,test_size=0.1,label='Survived')
y_train = np.array([y_train, -(y_train-1)]).T
y_test = np.array([y_test, -(y_test-1)]).T
X_train = X_train
y_train = y_train
X_test = X_test
y_test = y_test
print('y',y_test.T[1])





print('len',len(y_test))


# Dimensions of dataset
n = X_train.shape[0]
p = titanicDF.shape[1]

print('number of instances: '+str(n))
print('number of features: '+str(p))

# Training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n









# Initializers
sigma = 1
weight_initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG", uniform=True)
bias_initializer = tf.zeros_initializer()

# Model architecture parameters
n_stocks = 9
n_neurons_1 = 64
n_neurons_2 = 64
n_neurons_3 = 32
n_neurons_4 = 4
n_target = 2
# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases

#W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
#bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))


# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None,2])


# Hidden layer
hidden_1 = tf.nn.sigmoid(tf.matmul(X, W_hidden_1))
hidden_2 = tf.nn.sigmoid(tf.matmul(hidden_1, W_hidden_3))
#hidden_3 = tf.nn.sigmoid(tf.matmul(hidden_2, W_hidden_3))
hidden_4 = tf.nn.sigmoid(tf.matmul(hidden_2, W_hidden_4))

# Output layer (must be transposed)
out = tf.matmul(hidden_4, W_out)
#logits = tf.layers.dense(out, 1, activation=None)

# Cost function
#mse = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, Y))
#mse = tf.nn.l2_loss(out-Y, name="squared_error_cost")

#mse=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(out), labels=Y))
#mse = tf.nn.l2_loss(out-Y,name="squared_error_cost")
mse = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=Y))

# Optimizer
opt = tf.train.AdamOptimizer(0.0001).minimize(mse)
predict_op = tf.argmax(out,1)
# Make Session
net = tf.Session()
# Run initializer
net.run(tf.global_variables_initializer())

def differences(a, b):
    if len(a) != len(b):
        raise ValueError("Lists of different length.")
    return sum(i != j for i, j in zip(a, b))


# Number of epochs and batch size
epochs = 50000
batch_size = 256

costfunction = []
for e in range(epochs):
    #print('epoch '+str(e))
    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    '''
    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        #total_batch = int(len(y_train)/batch_size)
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch

        cost,_ = net.run([mse,opt], feed_dict={X: batch_x, Y: batch_y})
    '''
    cost,_ = net.run([mse,opt], feed_dict={X: X_train, Y: y_train})

    print(cost)

    #prediction = net.run(predict_op, feed_dict={X: X_test})
    #print(e)
    #print(y_test.T[0])
    #res = [abs(a_i - b_i) for a_i, b_i in zip(y_test.T[0], prediction)]\
    #diffs = differences(y_test.T[1], prediction)
    print(e, np.mean(np.argmax(y_test, axis=1) ==
                         net.run(predict_op, feed_dict={X: X_test, Y: y_test})))

    costfunction.append(cost)
    #print(pred)
print(net.run(predict_op, feed_dict={X: X_test, Y: y_test}))
plt.plot(costfunction)
plt.show()
        # Show progress
        #if np.mod(i, 5) == 0:
            # Prediction
            #plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            #file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.jpg'
            #plt.savefig(file_name)
            #plt.pause(0.01)


