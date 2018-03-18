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

import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#
PCA_FEATURE_SIZE = 5+1
SCALE_SIZE = 4
BALL_NUMBER = 7
FEATURE_SIZE = SCALE_SIZE * BALL_NUMBER * PCA_FEATURE_SIZE
CLASS_SIZE = 12

# target_names = { # 0: 'unknown',
#                1: 'Facade',
#                2: 'Ground',
#                3: 'Cars',
#                4: 'Moto',
#                5: 'Traffic signs',
#                6: 'Pedestrians'}

target_names = { # 0: 'unknown',
                1: 'ground',
                2: 'building',
                3: 'bollard',
                4: 'floor lamp',
                5: 'traffic light',
                6: 'sign',
                7: 'roasting',
                8: 'wire',
                9: '4+ wheels',
                10: 'trash can',
                11: 'natural'}

training_caches = ['../cache_Lille1_part1.npz']
training_filenames = ['../data/leman2/Lille1_partA.ply']

test_pt_cloud_file = '../data/leman2/check_Lille1_partB.ply'
test_cache_file = '../cache_test.npz'
test_file = '../data/leman2/Lille1_partB.ply'

def load_pts(file):
    p = PlyData.read(file)
    pts = np.vstack((p['vertex']['x'], p['vertex']['y'], p['vertex']['z'])).T
    return pts


def load_pts_label(file):
    p = PlyData.read(file)
    pts = np.vstack((p['vertex']['x'], p['vertex']['y'], p['vertex']['z'])).T
    labels = np.array(p['class'])
    return pts, labels


def load_features_label(file):
    p = PlyData.read(file)
    feats = p['feature']['entries']
    labels = p['feature']['class']
    features = np.array(list(feats))
    return features, labels


def load_features(file):
    p = PlyData.read(file)
    feats = p['feature']['entries']
    features = np.array(list(feats))
    return features


def advanced_point_choice(num_per_class_global):
    # Initiate arrays
    training_features = np.empty((0, FEATURE_SIZE))
    training_labels = np.empty((0,))

    # Loop over each training cloud
    for i in range(len(training_filenames)):
        # Load Training cloud
        feature_cache = training_caches[i]
        if exists(feature_cache):
            with Timer('!!Read from last Cache!!'):
                f = np.load(feature_cache)
                features = f['features']
                labels = f['labels']
        else:
            with Timer('reading features'):
                filename = training_filenames[i]
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
            if label != 0:
                min_n = min(np.sum(labels == label), min_n)

        if num_per_class > min_n:
            # num_per_class = min_n
            print("Minimum number of sample per class is only {}".format(min_n))

        # Loop over each class to choose training points
        for label in range(CLASS_SIZE):
            if label == 0:
                continue  # unclassified points

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
    a = np.arange(training_labels.shape[0])
    np.random.shuffle(a)
    return training_features[a], training_labels[a]


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


def mlp(useGPU, X, y, y_weight, test_X, test_Y, test_Y_weight):
    # Train a MLP
    # ********************************
    #
    device = '/device:GPU:0' if useGPU else '/device:CPU:0'
    with tf.device(device):
        model = Sequential()
        model.add(Dense(30, activation='relu', input_shape=(FEATURE_SIZE,)))
        # model.add(Dense(10, activation='relu'))
        model.add(Dense(CLASS_SIZE, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        #print_weight_count(model)
        model.summary()

    with Timer('Training MLP'):
        y_onehot = np_utils.to_categorical(y, CLASS_SIZE)  # one hot
        test_Y_one_hot = np_utils.to_categorical(test_Y, CLASS_SIZE)
        history = model.fit(X, y_onehot, epochs=100,
                            batch_size=2048, verbose=2,
                            # validation_split=0.1,
                            validation_data=(test_X, test_Y_one_hot, test_Y_weight),
                            sample_weight=y_weight)

    with Timer('Test'):
        y_prob = model.predict(test_X)
        predictions = y_prob.argmax(axis=1)
        confidence = y_prob.max(axis=1)
        print(classification_report(test_Y, predictions, digits=3,
                                    sample_weight=test_Y_weight,
                                    target_names=list(target_names.values())))
    return predictions, confidence, history


def mlp_conv(useGPU, X, y, y_weight, test_X, test_Y, test_Y_weight):
    # Train a MLP
    # ********************************
    #
    device = '/device:GPU:0' if useGPU else '/device:CPU:0'
    with tf.device(device):
        model = Sequential()
        model.add(Reshape((SCALE_SIZE * BALL_NUMBER, PCA_FEATURE_SIZE, 1), input_shape=(FEATURE_SIZE,)))

        # two layer 1d conv on pca feature for each ball
        nconv1 = 20
        model.add(Conv2D(nconv1, (1, PCA_FEATURE_SIZE), activation='relu'))
        model.add(Reshape((SCALE_SIZE * BALL_NUMBER, nconv1, 1)))
        # nconv2 = 10
        # model.add(Conv2D(nconv2, (1, nconv1), activation='relu'))
        # model.add(Reshape((SCALE_SIZE * BALL_NUMBER, nconv2, 1)))
        nconv3 = 10
        model.add(Conv2D(nconv3, (1, nconv1), activation='relu'))
        model.add(Reshape((SCALE_SIZE, BALL_NUMBER, -1)))

        model.add(Conv2D(5, (1, BALL_NUMBER), activation='relu'))
        model.add(Flatten())
        # model.add(Dense(10, activation='relu'))
        model.add(Dense(CLASS_SIZE, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        # print_weight_count(model)
        model.summary()

    with Timer('Training MLP'):
        y_onehot = np_utils.to_categorical(y, CLASS_SIZE)  # one hot
        test_Y_one_hot = np_utils.to_categorical(test_Y, CLASS_SIZE)
        history = model.fit(X, y_onehot, epochs=100,
                            batch_size=2048, verbose=2,
                            # validation_split=0.1,
                            validation_data=(test_X, test_Y_one_hot, test_Y_weight),
                            sample_weight=y_weight)

    with Timer('Test'):
        y_prob = model.predict(test_X)
        predictions = y_prob.argmax(axis=1)
        confidence = y_prob.max(axis=1)
        print(classification_report(test_Y, predictions, digits=3,
                                    sample_weight=test_Y_weight,
                                    target_names=list(target_names.values())))
    return predictions, confidence, history


def random_forest(X, y, y_weight, test_X, test_Y, test_Y_weight):
    X_train, X_valid, y_train, y_valid, y_weight_train, y_weight_valid = \
        train_test_split(X, y, y_weight, test_size=0.1, random_state=42)
    with Timer('Training Random Forest'):
        clf = RandomForestClassifier(n_estimators=50, max_depth=30)
        clf.fit(X_train, y_train, sample_weight=y_weight_train)

    with Timer('Random Forest validation'):
        y_pred = clf.predict(X_valid)
        print(classification_report(y_valid, y_pred, digits=3,
                                    sample_weight=y_weight_valid,
                                    target_names=list(target_names.values())))

    with Timer('Test'):
        predictions = clf.predict(test_X)
        print(classification_report(test_Y, predictions, digits=3,
                                    sample_weight=test_Y_weight,
                                    target_names=list(target_names.values())))
    return predictions


def training_and_test(useGPU=True, useMLP=True, useMLP_Conv=True, useRF=True):
    num_per_class = 10000


    # Collect training features / labels
    # **********************************
    #

    with Timer('Collect Training Features'):
        X, Y = advanced_point_choice(num_per_class)
        Y=Y.astype(int)
        Y_weight = class_weight.compute_sample_weight('balanced', Y)

    # Load cloud as a [N x 3] matrix
    with Timer('Collect Testing Features'):

        test_pt_cloud = load_pts(test_pt_cloud_file)
        if exists(test_cache_file):
            with Timer('!!Read from last Cache!!'):
                f = np.load(test_cache_file)
                test_X = f['features']
                test_Y = f['labels']
        else:
            with Timer('reading features'):
                test_X, test_Y = load_features_label(test_file)
                np.savez(test_cache_file, features=test_X, labels=test_Y)
        if not np.isfinite(test_X).all():
            print("feature_test contains nan or inf!")
            exit()
        test_Y = test_Y.astype(int)
        test_Y_weight = class_weight.compute_sample_weight('balanced', test_Y)

    his_train = np.bincount(Y, minlength=CLASS_SIZE)
    his_test = np.bincount(test_Y, minlength=CLASS_SIZE)
    print('Number of sample per class:')
    print('{:<20}\t{:>10}\t{:>10}'.format('class','train','test'))
    for i in range(1,CLASS_SIZE):
        print('{:<20}\t{:>10}\t{:>10}'.format(target_names[i], his_train[i], his_test[i]))

    if useRF:
        predictions = random_forest(X, Y, Y_weight, test_X, test_Y, test_Y_weight)
        with Timer('Save predictions ply'):
            write_ply('../preds_RF.ply',
                      [test_pt_cloud, predictions.astype(np.uint8)], ['x', 'y', 'z', 'labels'])

    if useMLP:
        predictions, confidence, mlp_history = mlp(useGPU, X, Y, Y_weight, test_X, test_Y, test_Y_weight)
        with Timer('Save predictions ply'):
            write_ply('../preds_MLP.ply',
                      [test_pt_cloud, predictions.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
            predictions_90 = predictions * (confidence > 0.9)
            write_ply('../preds_MLP_90.ply',
                      [test_pt_cloud, predictions_90.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
            predictions_95 = predictions * (confidence > 0.95)
            write_ply('../preds_MLP_95.ply',
                      [test_pt_cloud, predictions_95.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
            predictions_99 = predictions * (confidence > 0.99)
            write_ply('../preds_MLP_99.ply',
                      [test_pt_cloud, predictions_99.astype(np.uint8)], ['x', 'y', 'z', 'labels'])

    if useMLP_Conv:
        predictions, confidence, mlp_conv_history = mlp_conv(useGPU, X, Y, Y_weight, test_X, test_Y, test_Y_weight)
        with Timer('Save predictions ply'):
            write_ply('../preds_MLPconv.ply',
                      [test_pt_cloud, predictions.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
            predictions_90 = predictions * (confidence > 0.9)
            write_ply('../preds_MLPconv_90.ply',
                      [test_pt_cloud, predictions_90.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
            predictions_95 = predictions * (confidence > 0.95)
            write_ply('../preds_MLPconv_95.ply',
                      [test_pt_cloud, predictions_95.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
            predictions_99 = predictions * (confidence > 0.99)
            write_ply('../preds_MLPconv_99.ply',
                      [test_pt_cloud, predictions_99.astype(np.uint8)], ['x', 'y', 'z', 'labels'])
    if mlp_conv_history is not None:
        plt.plot(mlp_conv_history.history['loss'], label='conv loss')
        plt.plot(mlp_conv_history.history['val_loss'], label='conv test loss')
    if mlp_history is not None:
        plt.plot(mlp_history.history['loss'], label='mlp loss')
        plt.plot(mlp_history.history['val_loss'], label='mlp test loss')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

if __name__ == '__main__':
    training_and_test()
