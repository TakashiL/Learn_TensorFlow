import urllib.request
import csv
import numpy as np
import tensorflow as tf

# parameter
DAILY_DATA_URL = "http://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getInnerFuturesDailyKLine?symbol=M0"
RAW_DATA_FILE = "./data/raw_data.csv"
TRAIN_FILE = "./data/train_data.csv"
TEST_FILE = "./data/test_data.csv"


def data_download():
    url = urllib.request.urlopen(DAILY_DATA_URL)
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


def read_single_data(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_list = list(reader)
        features = []
        labels = []
        for item in data_list:
            item_feature = list(map(float, item[:6]))
            item_feature = list(map(int, item_feature))
            item_label = int(item[6])
            features.append(item_feature)
            labels.append(item_label)

    features = np.asarray(features)
    labels = np.asarray(labels)
    return features, labels


def read_data():
    train_features, train_labels = read_single_data(TRAIN_FILE)
    test_features, test_labels = read_single_data(TEST_FILE)
    return train_features, train_labels, test_features, test_labels


def data_read_contrib():
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=TRAIN_FILE, target_dtype=np.int, features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=TEST_FILE, target_dtype=np.int, features_dtype=np.float32)
    return training_set, test_set


if __name__ == '__main__':
    data_download()

