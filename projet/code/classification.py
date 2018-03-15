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
CLASS_SIZE = 13

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
    feature_caches = ['../feature_cache_1.npz']
    filenames = ['../data/feature_area_3.ply']
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
        for label, name in label_names.items():
            if label != 0:
                n = np.sum(labels == label)
                min_n = min(n, min_n)

        if num_per_class > min_n:
            # num_per_class = min_n
            print("Minimum number of sample per class is only {}".format(min_n))

        # Loop over each class to choose training points
        for label, name in label_names.items():
            # Do not include class 0 in training
            if label == 0:
                continue

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


def mlp(useGPU, X, y, y_weight, challengeX, challengeY):
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
        history = model.fit(X, y_onehot, epochs=50,
                            batch_size=2048, verbose=2, validation_split=0.1, sample_weight=y_weight)

    with Timer('Test'):
        y_prob = model.predict(challengeX)
        predictions = y_prob.argmax(axis=1)
        confidence = y_prob.max(axis=1)
        print(classification_report(challengeY, predictions, target_names=target_names))
    return predictions, confidence


def mlp_conv(useGPU, X, y, y_weight, challengeX, challengeY):
    # Train a MLP
    # ********************************
    #
    device = '/device:GPU:0' if useGPU else '/device:CPU:0'
    with tf.device(device):
        model = Sequential()
        model.add(Reshape((SCALE_SIZE * BALL_NUMBER, PCA_FEATURE_SIZE, 1), input_shape=(FEATURE_SIZE,)))
        model.add(Conv2D(8, (1, PCA_FEATURE_SIZE), activation='relu'))
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
        history = model.fit(X, y_onehot, epochs=50,
                            batch_size=2048, verbose=2, validation_split=0.1, sample_weight=y_weight)

    with Timer('Test'):
        y_prob = model.predict(challengeX)
        predictions = y_prob.argmax(axis=1)
        confidence = y_prob.max(axis=1)
        print(classification_report(challengeY, predictions, target_names=target_names))
    return predictions, confidence


def random_forest(X, y, y_weight, challengeX, challengeY):
    X_train, X_test, y_train, y_test, y_weight_train, _ = \
        train_test_split(X, y, y_weight, test_size=0.1, random_state=42)
    with Timer('Training Random Forest'):
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train, sample_weight=y_weight_train)

    with Timer('Random Forest validation'):
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=target_names))

    with Timer('Test'):
        predictions = clf.predict(challengeX)
        print(classification_report(challengeY, predictions, target_names=target_names))
    return predictions


def training_and_test(useGPU=True, useMLP=True, useMLP_Conv=True, useRF=True):
    num_per_class = 2000
    # label_names = {0: 'Unclassified',
    #                1: 'Ground',
    #                2: 'Building',
    #                3: 'Vegetation',
    #                4: 'Barriers',
    #                5: 'Cars',
    #                6: 'Signage'}

    label_names = {0: 'Unclassified',
                   1: 'C1',
                   2: 'C2',
                   3: 'C3',
                   4: 'C4',
                   5: 'C5',
                   6: 'C6',
                   7: 'C7',
                   8: 'C8',
                   9: 'C9',
                   10: 'C10',
                   11: 'C11',
                   12: 'C12'}

    # Collect training features / labels
    # **********************************
    #

    with Timer('Collect Training Features'):
        X, y = advanced_point_choice(num_per_class, label_names)
        y_weight = class_weight.compute_sample_weight('balanced', y)

    # Load cloud as a [N x 3] matrix
    # points_challenge = load_pts('../data/Lille_street_test.ply')
    points_challenge = load_pts('../data/checkfeature_area_4.ply')
    with Timer('Collect Testing Features'):
        challenge_cache = '../feature_test_cache.npz'
        if exists(challenge_cache):
            with Timer('!!Read from last Cache!!'):
                challengeX, challengeY = np.load(challenge_cache)
        else:
            with Timer('reading ply'):
                challengeX, challengeY = load_features_label('../data/feature_area_4.ply')
                np.savez(challenge_cache, challengeX=challengeX, challengeY=challengeY)
        if not np.isfinite(challengeX).all():
            print("feature_test contains nan or inf!")
            exit()

    if useRF:
        predictions = random_forest(X, y, y_weight, challengeX, challengeY)

        with Timer('Save predictions'):
            np.savetxt('../Lille_street_test_preds_RF.txt', predictions, fmt='%d')

        with Timer('Save predictions ply'):
            write_ply('../Lille_street_test_preds_RF.ply',
                      [points_challenge, predictions.astype(np.uint8)], ['x', 'y', 'z', 'labels'])

    if useMLP:
        predictions, confidence = mlp(useGPU, X, y, y_weight, challengeX, challengeY)
        with Timer('Save predictions'):
            np.savetxt('../Lille_street_test_preds_MLP.txt', predictions, fmt='%d')

        with Timer('Save predictions ply'):
            write_ply('../Lille_street_test_preds_MLP.ply',
                      [points_challenge, predictions.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
            predictions_90 = predictions * (confidence > 0.9)
            write_ply('../Lille_street_test_preds_MLP_90.ply',
                      [points_challenge, predictions_90.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
            predictions_95 = predictions * (confidence > 0.95)
            write_ply('../Lille_street_test_preds_MLP_95.ply',
                      [points_challenge, predictions_95.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
            predictions_99 = predictions * (confidence > 0.99)
            write_ply('../Lille_street_test_preds_MLP_99.ply',
                      [points_challenge, predictions_99.astype(np.uint8)], ['x', 'y', 'z', 'labels'])

    if useMLP_Conv:
        predictions, confidence = mlp_conv(useGPU, X, y, y_weight, challengeX, challengeY)
        with Timer('Save predictions'):
            np.savetxt('../Lille_street_test_preds_MLPconv.txt', predictions, fmt='%d')

        with Timer('Save predictions ply'):
            write_ply('../Lille_street_test_preds_MLPconv.ply',
                      [points_challenge, predictions.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
            predictions_90 = predictions * (confidence > 0.9)
            write_ply('../Lille_street_test_preds_MLPconv_90.ply',
                      [points_challenge, predictions_90.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
            predictions_95 = predictions * (confidence > 0.95)
            write_ply('../Lille_street_test_preds_MLPconv_95.ply',
                      [points_challenge, predictions_95.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
            predictions_99 = predictions * (confidence > 0.99)
            write_ply('../Lille_street_test_preds_MLPconv_99.ply',
                      [points_challenge, predictions_99.astype(np.uint8)], ['x', 'y', 'z', 'labels'])


if __name__ == '__main__':
    training_and_test()
