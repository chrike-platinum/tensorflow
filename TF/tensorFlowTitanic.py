__author__ = 'christiaan'

# coding: utf-8
import pandas as pd
import tensorflow as tf

CATEGORICAL_COLUMNS = ["Name", "Sex", "Embarked", "Cabin"]
CONTINUOUS_COLUMNS = ["Age", "SibSp", "Parch", "Fare", "PassengerId", "Pclass"]

SURVIVED_COLUMN = "Survived"


def cleanData(train,test):
    train = pd.read_csv(train)
    test = pd.read_csv(test)

    # Impute the missing ages with median age
    train["Age"] = train["Age"].fillna(train["Age"].median()).astype(int)
    test["Age"] = test["Age"].fillna(test["Age"].median()).astype(int)

    # Fill in missing embarked with S
    train["Embarked"] = train["Embarked"].fillna("S")
    test["Embarked"] = test["Embarked"].fillna("S")

    # Fill in missing Cabin with None
    train["Cabin"] = train["Cabin"].fillna("None")
    test["Cabin"] = test["Cabin"].fillna("None")

    # Write our changed dataframes to csv.
    test.to_csv("./test.csv", index=False)
    train.to_csv('./train.csv', index=False)
    return test,train


test,train = cleanData('/Users/christiaan/Desktop/train.csv','/Users/christiaan/Desktop/test.csv')


def transformToFeatureColumns(test):
    print(test)





def build_estimator(model_dir):
  """Build an estimator."""
  # Categorical columns
  sex = tf.contrib.layers.sparse_column_with_keys(column_name="Sex",
                                                     keys=["female", "male"])
  embarked = tf.contrib.layers.sparse_column_with_keys(column_name="Embarked",
                                                   keys=["C",
                                                         "S",
                                                         "Q"])

  cabin = tf.contrib.layers.sparse_column_with_hash_bucket(
      "Cabin", hash_bucket_size=1000)
  name = tf.contrib.layers.sparse_column_with_hash_bucket(
      "Name", hash_bucket_size=1000)


  # Continuous columns
  age = tf.contrib.layers.real_valued_column("Age")
  passenger_id = tf.contrib.layers.real_valued_column("PassengerId")
  sib_sp = tf.contrib.layers.real_valued_column("SibSp")
  parch = tf.contrib.layers.real_valued_column("Parch")
  fare = tf.contrib.layers.real_valued_column("Fare")
  p_class = tf.contrib.layers.real_valued_column("Pclass")

  # Transformations.
  age_buckets = tf.contrib.layers.bucketized_column(age,
                                                    boundaries=[
                                                        5, 18, 25, 30, 35, 40,
                                                        45, 50, 55, 65
                                                    ])
   # Wide columns and deep columns.
  wide_columns = [sex, embarked, cabin, name, age_buckets,
                  tf.contrib.layers.crossed_column(
                      [age_buckets, sex],
                      hash_bucket_size=int(1e6)),
                  tf.contrib.layers.crossed_column([embarked, name],
                                                   hash_bucket_size=int(1e4))]
  deep_columns = [
      tf.contrib.layers.embedding_column(sex, dimension=8),
      tf.contrib.layers.embedding_column(embarked, dimension=8),
      tf.contrib.layers.embedding_column(cabin, dimension=8),
      tf.contrib.layers.embedding_column(name, dimension=8),
      age,
      passenger_id,
      sib_sp,
      parch,
      fare,
      p_class
  ]



  return tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100,50,40])

def input_fn(df, train=False):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor( indices=[[i, 0] for i in range(df[k].size)],
    values=df[k].values, dense_shape=[df[k].size, 1]) for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  if train:
    label = tf.constant(df[SURVIVED_COLUMN].values)
      # Returns the feature columns and the label.
    print(feature_cols)
    return feature_cols, label
  else:
    return feature_cols


def train_and_eval():
  """Train and evaluate the model."""
  df_train = pd.read_csv(
      tf.gfile.Open("./train.csv"),
      skipinitialspace=True)
  df_test = pd.read_csv(
      tf.gfile.Open("./test.csv"),
      skipinitialspace=True)

  model_dir = "./models"
  print("model directory = %s" % model_dir)

  m = build_estimator(model_dir)
  m.fit(input_fn=lambda: input_fn(df_train, True), steps=200)
  print(list(m.predict(input_fn=lambda: input_fn(df_test))))
  results = m.evaluate(input_fn=lambda: input_fn(df_train, True), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))

def main(_):
  train_and_eval()


if __name__ == "__main__":
  tf.app.run()
