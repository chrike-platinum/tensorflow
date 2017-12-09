__author__ = 'christiaan'

import tensorflow as tf
import pandas as pd

titanicDF = pd.read_csv('/Users/christiaan/Downloads/ML/Exercice/titanic.csv',na_values='NaN')
print(titanicDF)


_CSV_COLUMNS = [
    'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex',
    'Age', 'SibSp', 'Parch', 'Ticket', 'Fare',
    'Cabin', 'Embarked']

_CSV_COLUMN_DEFAULTS=[[0], [0], [0], [''], [''], [0], [0], [0], [''], [0],
                        [''], ['']]
_SHUFFLE_BUFFER=4
#dataset = tf.data.TextLineDataset('/Users/christiaan/Downloads/ML/Exercice/titanic.csv')

def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have either run data_download.py or '
      'set both arguments --train_data and --test_data.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('Survived')
    return features, labels

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels


tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)


print(input_fn('/Users/christiaan/Downloads/ML/Exercice/titanic.csv',100,False,128))