import csv
import numpy as np

TRAIN_FILE = "./data/train_data.csv"
TEST_FILE = "./data/test_data.csv"


def read_single_data(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_list = list(reader)
        features = []
        labels = []
        for item in data_list:
            item_feature = list(map(float, item[:5]))
            item_label = float(item[5])
            if item_label > 0.5:  # raise
                item_label = 2
            elif item_label < -0.5:  # fall
                item_label = 0
            else:
                item_label = 1  # same
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
