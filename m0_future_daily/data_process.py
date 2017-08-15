import urllib.request
import csv
import numpy as np
import tensorflow as tf

# parameter
DAILY_DATA_URL = "http://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getInnerFuturesDailyKLine?symbol=M0"
RAW_TOTAL_DATA_FILE = "./data/raw_total_data.csv"
RAW_TRAIN_FILE = "./data/raw_train_data.csv"
RAW_TEST_FILE = "./data/raw_test_data.csv"
TOTAL_FILE = "./data/total_data.csv"
TRAIN_FILE = "./data/train_data.csv"
TEST_FILE = "./data/test_data.csv"
LINEAR_TRAIN = "./lineardata/train_data.csv"
LINEAR_TEST = "./lineardata/test_data.csv"


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

    with open(RAW_TOTAL_DATA_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in data:
            writer.writerow(line)
        print("wrote data successfully: " + RAW_TOTAL_DATA_FILE)

    with open(RAW_TRAIN_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in data[:2926]:
            writer.writerow(line)
        print("wrote data successfully: " + RAW_TRAIN_FILE)

    with open(RAW_TEST_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in data[2926:]:
            writer.writerow(line)
        print("wrote data successfully: " + RAW_TEST_FILE)


def data_extract():
    trans_data = []

    with open(RAW_TOTAL_DATA_FILE, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_list = list(reader)
        for index in range(4, len(data_list)):  # 2500 train data
            new_line = [data_list[index-4][4],
                        data_list[index-3][4],
                        data_list[index-2][4],
                        data_list[index-1][4],
                        data_list[index][4],
                        data_list[index][6]]
            trans_data.append(new_line)

    with open(TOTAL_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in trans_data:
            writer.writerow(line)
        print("wrote data successfully: " + TOTAL_FILE)

    with open(TRAIN_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in trans_data[:2800]:
            writer.writerow(line)
        print("wrote data successfully: " + TRAIN_FILE)

    with open(TEST_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in trans_data[2800:3000]:
            writer.writerow(line)
        print("wrote data successfully: " + TEST_FILE)


def create_linear_data():
    with open(TOTAL_FILE, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_list = list(reader)
        for index in range(len(data_list)-1):
            todayprice = float(data_list[index][4])
            tmrprice = float(data_list[index+1][4])
            newout = 1.0 * (tmrprice - todayprice) / todayprice * 100
            data_list[index][5] = newout

    with open(LINEAR_TRAIN, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in data_list[:2800]:
            writer.writerow(line)
        print("wrote data successfully: " + LINEAR_TRAIN)

    with open(LINEAR_TEST, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in data_list[2800:3000]:
            writer.writerow(line)
        print("wrote data successfully: " + LINEAR_TEST)


def read_single_data(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_list = list(reader)
        features = []
        labels = []
        for item in data_list:
            item_feature = list(map(float, item[:5]))
            item_label = int(item[5])
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


def next_batch(num, features, labels):
    """
    Args:
        num: Size of next batch
        features: Features of data
        labels: Labels of data
    Returns:
        next_features: features of next batch
        next_labels: labels of next batch
    """
    idx = np.arange(0, len(features))
    np.random.shuffle(idx)
    idx = idx[0:num]
    feature_shuffle = [features[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    next_features = np.asarray(feature_shuffle)
    next_labels = np.asarray(labels_shuffle)
    return next_features, next_labels


if __name__ == '__main__':
    create_linear_data()
