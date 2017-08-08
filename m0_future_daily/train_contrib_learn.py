import os
import sys
import argparse
from m0_future_daily.data_process import data_read_contrib
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def build_classifier():
    # Sepcify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=6)]
    # Build 3 layer DNN with specified units respectively.
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[FLAGS.hidden1, FLAGS.hidden2, FLAGS.hidden3],
        n_classes=2,
        model_dir=FLAGS.log_dir)
    return classifier


def input_fn(data_set):
    feature = tf.constant(data_set.data)
    label = tf.constant(data_set.target)
    return feature, label


def run_training():
    training_set, test_set = data_read_contrib()

    classifier = build_classifier()

    # Fit model
    classifier.fit(input_fn=lambda: input_fn(training_set), steps=FLAGS.max_steps)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=lambda: input_fn(test_set), steps=1)["accuracy"]

    print("\nTest Accuracy: %f\n" % accuracy_score)

    predicts = list(classifier.predict_classes(input_fn=lambda: input_fn_predict(test_set)))
    print(predicts)


def input_fn_predict(data_set):
    feature = tf.constant(data_set.data)
    return feature, None


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=20000, help='Number of steps to run trainer.')
    parser.add_argument('--hidden1', type=int, default=20, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=20, help='Number of units in hidden layer 2.')
    parser.add_argument('--hidden3', type=int, default=20, help='Number of units in hidden layer 3.')
    parser.add_argument('--log_dir', type=str, default='.\savedir/DNN_model', help='Directory to put the log data.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)