__author__ = 'christiaan'
import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from sklearn.model_selection import train_test_split,cross_val_score
import pandas as pd

# Import and transform dataset
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
titanicDF['AgeGroup'] = titanicDF[['Age']].applymap(replaceAgeGroups)


titanicDF['FareGroup'] = titanicDF[['Fare']].applymap(replaceFareGroups)
titanicDF=titanicDF.drop(['Fare'], axis=1)



titanicDF['groupSize'] = titanicDF['SibSp'] + titanicDF['Parch'] + 1

titanicDF['Alone']= np.where((titanicDF['groupSize']==1),1,0)



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

#
#titanicDF.drop(['PassengerId'],axis=1,inplace=True)
titanicDF.drop(['PassengerId'], axis=1,inplace=True)

#reset index df
titanicDF=titanicDF.apply(pd.to_numeric)
titanicDF.reset_index(inplace=True)
titanicDF = normalize(titanicDF, axis=1, norm='max')
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

print(X_test)
print(Y_test)


# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 100
display_step = 1


# Network Parameters
n_hidden_1 = 16 # 1st layer number of features
n_hidden_2 = 4 # 2nd layer number of features
n_input = 12 # Number of feature
n_classes = 2 # Number of classes to predict


# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    out_layer = tf.nn.softmax(out_layer)
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X)/batch_size)
        X_batches = np.array_split(X, total_batch)
        Y_batches = np.array_split(Y, total_batch)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    print(sess.run(pred,feed_dict={x: X_test}))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))
    global result
    result = tf.argmax(pred, 1).eval({x: X_test, y: Y_test})