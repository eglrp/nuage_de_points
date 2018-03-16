#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
from utils.withtimer import Timer

from plyfile import PlyData, PlyElement

from os.path import exists

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#
PCA_FEATURE_SIZE = 6 + 1
SCALE_SIZE = 5
BALL_NUMBER = 7
FEATURE_SIZE = SCALE_SIZE * BALL_NUMBER * PCA_FEATURE_SIZE
CLASS_SIZE = 6

target_names = ['C{}'.format(i) for i in range(CLASS_SIZE)]


def load_pts(file):
    p = PlyData.read(file)
    pts = np.vstack((p['vertex']['x'], p['vertex']['y'], p['vertex']['z'])).T
    return pts


def load_pts_label(file):
    p = PlyData.read(file)
    pts = np.vstack((p['vertex']['x'], p['vertex']['y'], p['vertex']['z'])).T
    labels = np.array(p['labels'])
    return pts, labels


def load_features_label(file):
    p = PlyData.read(file)
    feats = p['feature']['entries']
    labels = p['feature']['label']
    features = np.array(list(feats))
    return features, labels


def load_features(file):
    p = PlyData.read(file)
    feats = p['feature']['entries']
    features = np.array(list(feats))
    return features


def advanced_point_choice(num_per_class_global, label_names):
    # Initiate arrays
    training_features = np.empty((0, FEATURE_SIZE))
    training_labels = np.empty((0,))

    # feature_caches = ['../feature_cache_1.npz', '../feature_cache_2.npz', '../feature_cache_3.npz']
    # filenames = ['../data/feature_1.ply', '../data/feature_2.ply', '../data/feature_3.ply']
    feature_caches = ['../madame_cache_1.npz']
    filenames = ['../data/madame_1.ply']
    # Loop over each training cloud
    for i in range(len(filenames)):
        # Load Training cloud
        feature_cache = feature_caches[i]
        if exists(feature_cache):
            with Timer('!!Read from last Cache!!'):
                f = np.load(feature_cache)
                features = f['features']
                labels = f['labels']
        else:
            with Timer('reading ply'):
                filename = filenames[i]
                features, labels = load_features_label(filename)
                np.savez(feature_cache, features=features, labels=labels)

        num_per_class = num_per_class_global
        if not np.isfinite(features).all() or not np.isfinite(labels).all():
            print("{} contains nan or inf!".format(filename))
            exit()
        # Initiate training indices array
        training_inds = np.empty(0, dtype=np.int32)

        min_n = 1e100
        for label in range(CLASS_SIZE):
            n = np.sum(labels == label)
            min_n = min(n, min_n)

        if num_per_class > min_n:
            # num_per_class = min_n
            print("Minimum number of sample per class is only {}".format(min_n))

        # Loop over each class to choose training points
        for label in range(CLASS_SIZE):


            # Collect all indices of the current class
            label_inds = np.where(labels == label)[0]

            # If you have not enough indices, just take all of them
            if len(label_inds) <= num_per_class:
                training_inds = np.hstack((training_inds, label_inds))

            # If you have more than enough indices, choose randomly
            else:
                random_choice = np.random.choice(len(label_inds), num_per_class, replace=False)
                training_inds = np.hstack((training_inds, label_inds[random_choice]))

        # Compute features for the points of the chosen indices
        selected_features = features[training_inds, :]

        # Concatenate features / labels of all clouds
        training_features = np.vstack((training_features, selected_features))
        training_labels = np.hstack((training_labels, labels[training_inds]))

    return training_features, training_labels


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Reshape
import keras.utils.np_utils as np_utils


def print_weight_count(model):
    ws = model.get_weights()
    total_size = 0
    for w in ws:
        print('{} => {}'.format(w.shape, w.size))
        total_size += w.size
    print('total_size = {}'.format(total_size))


def mlp(useGPU, X, y, y_weight, test_X, test_Y):
    # Train a MLP
    # ********************************
    #
    device = '/device:GPU:0' if useGPU else '/device:CPU:0'
    with tf.device(device):
        model = Sequential()
        model.add(Dense(50, activation='relu', input_shape=(FEATURE_SIZE,)))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(CLASS_SIZE, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        print_weight_count(model)

    with Timer('Training MLP'):
        y_onehot = np_utils.to_categorical(y, CLASS_SIZE)  # one hot
        test_Y_one_hot = np_utils.to_categorical(test_Y, CLASS_SIZE)
        history = model.fit(X, y_onehot, epochs=10,
                            batch_size=2048, verbose=2,
                            # validation_split=0.1,
                            validation_data=(test_X, test_Y_one_hot),
                            sample_weight=y_weight)

    with Timer('Test'):
        y_prob = model.predict(test_X)
        predictions = y_prob.argmax(axis=1)
        confidence = y_prob.max(axis=1)
        print(classification_report(test_Y, predictions, target_names=target_names))
    return predictions, confidence


def mlp_conv(useGPU, X, y, y_weight, test_X, test_Y):
    # Train a MLP
    # ********************************
    #
    device = '/device:GPU:0' if useGPU else '/device:CPU:0'
    with tf.device(device):
        model = Sequential()
        model.add(Reshape((SCALE_SIZE * BALL_NUMBER, PCA_FEATURE_SIZE, 1), input_shape=(FEATURE_SIZE,)))

        # two layer 1d conv on pca feature for each ball
        model.add(Conv2D(8, (1, PCA_FEATURE_SIZE), activation='relu'))
        model.add(Reshape((SCALE_SIZE * BALL_NUMBER, 8, 1)))
        model.add(Conv2D(8, (1, 8), activation='relu'))
        model.add(Reshape((SCALE_SIZE, BALL_NUMBER, -1)))

        model.add(Conv2D(8, (1, BALL_NUMBER), activation='relu'))
        model.add(Flatten())
        model.add(Dense(20, activation='relu'))
        model.add(Dense(CLASS_SIZE, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        print_weight_count(model)

    with Timer('Training MLP'):
        y_onehot = np_utils.to_categorical(y, CLASS_SIZE)  # one hot
        test_Y_one_hot = np_utils.to_categorical(test_Y, CLASS_SIZE)
        history = model.fit(X, y_onehot, epochs=100,
                            batch_size=2048, verbose=2,
                            #validation_split=0.1,
                            validation_data=(test_X, test_Y_one_hot),
                            sample_weight=y_weight)

    with Timer('Test'):
        y_prob = model.predict(test_X)
        predictions = y_prob.argmax(axis=1)
        confidence = y_prob.max(axis=1)
        print(classification_report(test_Y, predictions, target_names=target_names))
    return predictions, confidence


def random_forest(X, y, y_weight, test_X, test_Y):
    X_train, X_test, y_train, y_test, y_weight_train, _ = \
        train_test_split(X, y, y_weight, test_size=0.1, random_state=42)
    with Timer('Training Random Forest'):
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train, sample_weight=y_weight_train)

    with Timer('Random Forest validation'):
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=target_names))

    with Timer('Test'):
        predictions = clf.predict(test_X)
        print(classification_report(test_Y, predictions, target_names=target_names))
    return predictions


def training_and_test(useGPU=True, useMLP=False, useMLP_Conv=True, useRF=True):
    num_per_class = 3000
    # label_names = {0: 'Unclassified',
    #                1: 'Ground',
    #                2: 'Building',
    #                3: 'Vegetation',
    #                4: 'Barriers',
    #                5: 'Cars',
    #                6: 'Signage'}

    label_names = {0: 'C0',
                   1: 'C1',
                   2: 'C2',
                   3: 'C3',
                   4: 'C4',
                   5: 'C5',}

    # Collect training features / labels
    # **********************************
    #

    with Timer('Collect Training Features'):
        X, y = advanced_point_choice(num_per_class, label_names)
        y_weight = class_weight.compute_sample_weight('balanced', y)

    # Load cloud as a [N x 3] matrix
    with Timer('Collect Testing Features'):
        pt_cloud_test = load_pts('../data/check_madame_2.ply')
        test_cache = '../madame_cache_2.npz'
        if exists(test_cache):
            with Timer('!!Read from last Cache!!'):
                f = np.load(test_cache)
                test_X = f['features']
                test_Y = f['labels']
        else:
            with Timer('reading ply'):
                filename = '../data/madame_2.ply'
                test_X, test_Y = load_features_label(filename)
                np.savez(test_cache, features=test_X, labels=test_Y)
        if not np.isfinite(test_X).all():
            print("feature_test contains nan or inf!")
            exit()

    if useRF:
        predictions = random_forest(X, y, y_weight, test_X, test_Y)
        with Timer('Save predictions ply'):
            write_ply('../preds_RF.ply',
                      [pt_cloud_test, predictions.astype(np.uint8)], ['x', 'y', 'z', 'labels'])

    if useMLP:
        predictions, confidence = mlp(useGPU, X, y, y_weight, test_X, test_Y)
        with Timer('Save predictions ply'):
            write_ply('../preds_MLP.ply',
                      [pt_cloud_test, predictions.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
            predictions_90 = predictions * (confidence > 0.9)
            write_ply('../preds_MLP_90.ply',
                      [pt_cloud_test, predictions_90.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
            predictions_95 = predictions * (confidence > 0.95)
            write_ply('../preds_MLP_95.ply',
                      [pt_cloud_test, predictions_95.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
            predictions_99 = predictions * (confidence > 0.99)
            write_ply('../preds_MLP_99.ply',
                      [pt_cloud_test, predictions_99.astype(np.uint8)], ['x', 'y', 'z', 'labels'])

    if useMLP_Conv:
        predictions, confidence = mlp_conv(useGPU, X, y, y_weight, test_X, test_Y)
        with Timer('Save predictions ply'):
            write_ply('../preds_MLPconv.ply',
                      [pt_cloud_test, predictions.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
            predictions_90 = predictions * (confidence > 0.9)
            write_ply('../preds_MLPconv_90.ply',
                      [pt_cloud_test, predictions_90.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
            predictions_95 = predictions * (confidence > 0.95)
            write_ply('../preds_MLPconv_95.ply',
                      [pt_cloud_test, predictions_95.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
            predictions_99 = predictions * (confidence > 0.99)
            write_ply('../preds_MLPconv_99.ply',
                      [pt_cloud_test, predictions_99.astype(np.uint8)], ['x', 'y', 'z', 'labels'])


if __name__ == '__main__':
    training_and_test()
