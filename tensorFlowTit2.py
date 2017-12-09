__author__ = 'christiaan'

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

train, test = train_test_split(data, test_size=0.001)

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

print("Dimensions of the training set : {0}".format(np.shape(X_train)))
print("Dimensions of the training set (target) : {0}".format(np.shape(y_train.values.reshape(len(y_train),1))))


def model(hu, model_dir, features):
    # Specify the shape of the features columns, so [5,1] here
    feature_columns = [tf.feature_column.numeric_column("x", shape=[len(features),1])]

    # Build n layer DNN with hu units (hu is an array)
    # The default optimizer is "AdaGrad" but we can specify another model
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=hu,
                                        n_classes=2,
                                        optimizer=tf.train.ProximalGradientDescentOptimizer(
                                            learning_rate=0.01,
                                            l1_regularization_strength=0.1,
                                            l2_regularization_strength=0.1),
                                        model_dir=model_dir)

    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(X_train)},
        y=np.array(y_train.values.reshape((len(y_train),1))),
        num_epochs=None,
        shuffle=True)
    return classifier, train_input_fn