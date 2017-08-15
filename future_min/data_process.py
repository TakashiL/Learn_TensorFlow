import glob
import csv
import numpy as np
import tensorflow as tf

TOTAL_FILE = "./dataag/total_data.csv"
TRAIN_FILE = "./dataag/train_data.csv"
TEST_FILE = "./dataag/test_data.csv"


def process_rawdata():
    path = r'D:\personal_folders\Ziyue_Lu\Learn_TensorFlow\future_min\data_ag'  # use your path
    allFiles = glob.glob(path + "/*.csv")
    trans_data = []
    for file_ in allFiles:
        print(file_)
        with open(file_, 'r') as csvfile:
            reader = csv.reader(csvfile)
            data_list = list(reader)
            for index in range(6, len(data_list)-1):
                new_line = [data_list[index - 4][5],
                            data_list[index - 3][5],
                            data_list[index - 2][5],
                            data_list[index - 1][5],
                            data_list[index][5]]
                if float(data_list[index][5]) < float(data_list[index + 1][5]):
                    new_line.append(2)  # closing price will rise tomorrow
                elif float(data_list[index][5]) > float(data_list[index + 1][5]):
                    new_line.append(0)  # closing price will fall tomorrow
                else:
                    new_line.append(1)  # closing price will be same with this min
                trans_data.append(new_line)

    length = len(trans_data)
    print(length)
    with open(TOTAL_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in trans_data:
            writer.writerow(line)
        print("wrote data successfully: " + TOTAL_FILE)

    with open(TRAIN_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in trans_data[:530000]:
            writer.writerow(line)
        print("wrote data successfully: " + TRAIN_FILE)

    with open(TEST_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in trans_data[530000:540000]:
            writer.writerow(line)
        print("wrote data successfully: " + TEST_FILE)


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
