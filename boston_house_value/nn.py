
import itertools
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# labels and feature names
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"

training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,skiprows=1, names=COLUMNS)
test_set = pd.read_csv("boston_test.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)

feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

regressor = tf.estimator.DNNClassifier(feature_columns=feature_cols,
                                       hidden_units=[10, 10],
                                       model_dir='/tmp/boston_model')
def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k:data_set[k].values for k in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        shuffle=shuffle,
        num_epochs=num_epochs
    )

regressor.train(input_fn=get_input_fn(training_set), steps=)