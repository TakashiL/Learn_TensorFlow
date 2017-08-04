import urllib.request
import csv
import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# data source and data sets
DATA_URL = "http://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getInnerFuturesDailyKLine?symbol=M0"
RAW_DATA_FILE = "raw_data.csv"
TRAIN_FILE = "train_data.csv"
TEST_FILE = "test_data.csv"


def data_process():
    url = urllib.request.urlopen(DATA_URL)
    data = url.read().decode()
    data = data.replace("[[", "")
    data = data.replace("]]", "")
    data = data.replace("\"", "")
    data = data.split("],[")
    for i in range(len(data)):
        data[i] = data[i].split(",")
    print("days: " + str(len(data)))

    index = 0
    for i in range(len(data) - 1):
        data[i][0] = index
        index += 1
        if float(data[i][4]) < float(data[i + 1][4]):
            data[i].append(1)  # closing price will rise tomorrow
        else:
            data[i].append(0)  # closing price will not rise tomorrow
    del data[-1]

    with open(RAW_DATA_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in data:
            writer.writerow(line)
        print("wrote data successfully: " + RAW_DATA_FILE)

    with open(TRAIN_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in data[:2926]:
            writer.writerow(line)
        print("wrote data successfully: " + TRAIN_FILE)

    with open(TEST_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in data[2926:]:
            writer.writerow(line)
        print("wrote data successfully: " + TEST_FILE)


def main():
    data_process()

    # Load datasets:
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=TRAIN_FILE, target_dtype=np.int, features_dtype=np.float)
    test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=TEST_FILE, target_dtype=np.int, features_dtype=np.float)

    # Sepcify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=6)]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=2, model_dir="/save/future_model")

    # Define the training inputs
    def get_train_inputs():
        x = tf.constant(training_set.data)
        y = tf.constant(training_set.target)
        return x, y

    # Fit model
    classifier.fit(input_fn=get_train_inputs, steps=2000)

    # Define the test inputs
    def get_test_inputs():
        x = tf.constant(test_set.data)
        y = tf.constant(test_set.target)
        return x, y

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]

    print("\nTest Accuracy: %f\n" % accuracy_score)

if __name__ == "__main__":
    main()