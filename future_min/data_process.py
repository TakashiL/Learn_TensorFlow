import glob
import csv
import pandas as pd

RAW_DATA_FILE = "raw_data.csv"
TRAIN_FILE = "train_data.csv"
TEST_FILE = "test_data.csv"

def process_rawdata():
    path = r'D:\personal_folders\Ziyue_Lu\Learn_TensorFlow\future_min\data_a'  # use your path
    allFiles = glob.glob(path + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        print(file_)
        df = pd.read_csv(file_, usecols=[1, 2, 3, 4, 5, 6, 7], skiprows=1, header=None)
        df = df[1:]
        df = df[:-1]
        list_.append(df)
    frame = pd.concat(list_)
    data = frame.values.tolist()

    for index in range(len(data) - 1):
        data[index][0] = index
        if float(data[index][4]) < float(data[index + 1][4]):
            data[index].append(2)  # closing price will rise tomorrow
        elif float(data[index][4]) > float(data[index + 1][4]):
            data[index].append(0)  # closing price will fall tomorrow
        else:
            data[index].append(1)  # closing price will be same with this min
    del data[-1]
    print("phase2")

    length = len(data)
    with open(RAW_DATA_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in data:
            writer.writerow(line)
        print("wrote data successfully: " + RAW_DATA_FILE)

    with open(TRAIN_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in data[:length-10000]:
            writer.writerow(line)
        print("wrote data successfully: " + TRAIN_FILE)

    with open(TEST_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in data[length-10000:]:
            writer.writerow(line)
        print("wrote data successfully: " + TEST_FILE)

process_rawdata()