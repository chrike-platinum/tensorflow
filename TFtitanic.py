__author__ = 'christiaan'
import tensorflow as tf
import numpy as np
import scipy.io
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
titanicDF.reset_index(inplace=True)


def splitTestTrain(df,test_size=0.1,label='Survived'):
    labels = df[label].values
    featuresL = df.drop(['Survived'],axis=1)
    features = df.drop(['Survived'],axis=1).values
    featuresNames = list(featuresL)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)
    return X_train, X_test, y_train, y_test,featuresNames

X_train, X_test, y_train, y_test,t = splitTestTrain(titanicDF,test_size=0.1,label='Survived')

y_train = np.array([y_train, -(y_train-1)]).T
y_test = np.array([y_test, -(y_test-1)]).T


# Split dataset into training (66%) and test (33%) set
training_set    = X_train.reshape(593,12)
print('XSHAPE',X_train.shape)
print('YSHAPE',y_test.shape)
training_labels = y_train.reshape(593,2)
test_set        = X_test
test_labels     = y_test.reshape(66,2)

print("Dataset ready.")

# Parameters
learning_rate   = 0.01 #argv
mini_batch_size = 100
training_epochs = 10000
display_step    = 500

# Network Parameters
n_hidden_1  = 64    # 1st hidden layer of neurons
n_hidden_2  = 16    # 2nd hidden layer of neurons
n_hidden_3  = 4    # 3rd hidden layer of neurons
n_input     = X_train.shape[1]   # number of features after LSA


# Tensorflow Graph input
x = tf.placeholder(tf.float64, shape=[None, n_input], name="x-data")
y = tf.placeholder(tf.float64, shape=[None, 2], name="y-labels")

print("Creating model.")

# Create model
def multilayer_perceptron(x, weights):
    # First hidden layer with SIGMOID activation
    layer_1 = tf.matmul(x, weights['h1'])
    layer_1 = tf.nn.relu(layer_1)
    # Second hidden layer with SIGMOID activation
    layer_2 = tf.matmul(layer_1, weights['h2'])
    layer_2 = tf.nn.relu(layer_2)
    # Third hidden layer with SIGMOID activation
    layer_3 = tf.matmul(layer_2, weights['h3'])
    layer_3 = tf.nn.relu(layer_3)
    # Output layer with SIGMOID activation
    out_layer = tf.matmul(layer_3, weights['out'])
    out_layer = tf.nn.softmax(out_layer)
    return out_layer

# Layer weights, should change them to see results
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], dtype=np.float64)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], dtype=np.float64)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3],dtype=np.float64)),
    'out': tf.Variable(tf.random_normal([n_hidden_3, 2], dtype=np.float64))
}

# Construct model
pred = multilayer_perceptron(x, weights)

# Define loss and optimizer
cost = tf.nn.l2_loss(pred-y,name="squared_error_cost")
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

print("Model ready.")

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    print("Starting Training.")

    # Training cycle
    for epoch in range(training_epochs):
        #avg_cost = 0.
        # minibatch loading
        minibatch_x = training_set[mini_batch_size*epoch:mini_batch_size*(epoch+1)]
        minibatch_y = training_labels[mini_batch_size*epoch:mini_batch_size*(epoch+1)]
        # Run optimization op (backprop) and cost op
        _, c = sess.run([optimizer, cost], feed_dict={x: minibatch_x, y: minibatch_y})

        # Compute average loss
        avg_cost = c / (minibatch_x.shape[0])

        # Display logs per epoch
        if (epoch) % display_step == 0:
            print("Epoch:", '%05d' % (epoch), "Training error=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    print('prediction',sess.run(pred,feed_dict={x:training_set}))

    test_error = tf.nn.l2_loss(pred-y,name="squared_error_test_cost")/test_set.shape[0]
    print("Test Error:", test_error.eval({x: test_set, y: test_labels}))

    #prediction:
