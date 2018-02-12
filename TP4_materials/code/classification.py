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

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply
from descriptors import local_PCA, compute_features, compute_features_advanced

# Import time package
import time

from os.path import exists

#Keras
import keras.utils.np_utils as np_utils
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#

def basic_point_choice(num_per_class, radius, label_names):

    # Initiate arrays
    training_features = np.empty((0, 4))
    training_labels = np.empty((0,))

    # Loop over each training cloud
    for i in range(1, 4):

        # Load Training cloud
        cloud_path = '../data/Lille_street_{:d}.ply'.format(i)
        cloud_ply = read_ply(cloud_path)
        points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        labels = cloud_ply['labels']

        # Initiate training indices array
        training_inds = np.empty(0, dtype=np.int32)

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
                random_choice = np.random.choice(len(label_inds), num_per_class , replace=False)
                training_inds = np.hstack((training_inds, label_inds[random_choice]))

        # Compute features for the points of the chosen indices
        verticality, linearity, planarity, sphericity = compute_features(points[training_inds, :], points, radius)
        features = np.vstack((verticality.ravel(), linearity.ravel(), planarity.ravel(), sphericity.ravel())).T

        # Concatenate features / labels of all clouds
        training_features = np.vstack((training_features, features))
        training_labels = np.hstack((training_labels, labels[training_inds]))

    return training_features, training_labels


def basic_training_and_test():

    # Parameters
    # **********
    #

    radius = 0.5
    num_per_class = 500
    label_names = {0: 'Unclassified',
                   1: 'Ground',
                   2: 'Building',
                   3: 'Vegetation',
                   4: 'Barriers',
                   5: 'Cars',
                   6: 'Signage'}

    # Collect training features / labels
    # **********************************
    #

    print('Collect Training Features')
    t0 = time.time()

    training_features, training_labels = basic_point_choice(num_per_class, radius, label_names)

    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))

    # Train a random forest classifier
    # ********************************
    #

    print('Training Random Forest')
    t0 = time.time()

    clf = RandomForestClassifier()
    clf.fit(training_features, training_labels)

    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))

    # Test
    # ****
    #

    print('Compute testing features')
    t0 = time.time()

    # Load cloud as a [N x 3] matrix
    cloud_path = '../data/Lille_street_test.ply'
    cloud_ply = read_ply(cloud_path)
    points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

    # Compute features only one time and save them for further use
    #
    #   WARNING : This will save you some time but do not forget to delete your features file if you change
    #             your features. Otherwise you will not compute them and use the previous ones
    #
    feature_file = '../data/Lille_street_test_features.npy'
    if exists(feature_file):
        features = np.load(feature_file)
    else:
        verticality, linearity, planarity, sphericity = compute_features(points, points, radius)
        features = np.vstack((verticality.ravel(), linearity.ravel(), planarity.ravel(), sphericity.ravel())).T
        np.save(feature_file, features)

    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))


    print('Test')
    t0 = time.time()
    predictions = clf.predict(features)
    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))

    print('Save predictions')
    t0 = time.time()
    np.savetxt('../data/Lille_street_test_preds.txt', predictions, fmt='%d')
    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))

    print('Save predictions ply')
    t0 = time.time()
    write_ply('../Lille_street_test_preds.ply',
              [points, predictions],
              ['x', 'y', 'z', 'labels'])
    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))

def advanced_point_choice(num_per_class, radius, label_names):

    # Initiate arrays
    training_features = np.empty((0, 28))
    training_labels = np.empty((0,))

    # Loop over each training cloud
    for i in range(1, 4):

        # Load Training cloud
        cloud_path = '../data/Lille_street_{:d}.ply'.format(i)
        cloud_ply = read_ply(cloud_path)
        points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        labels = cloud_ply['labels']

        # Initiate training indices array
        training_inds = np.empty(0, dtype=np.int32)

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
                random_choice = np.random.choice(len(label_inds), num_per_class , replace=False)
                training_inds = np.hstack((training_inds, label_inds[random_choice]))

        # Compute features for the points of the chosen indices
        features = compute_features_advanced(points[training_inds, :], points, radius)

        # Concatenate features / labels of all clouds
        training_features = np.vstack((training_features, features))
        training_labels = np.hstack((training_labels, labels[training_inds]))

    return training_features, training_labels

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.optimizers import Adam

def deep_training_and_test(useGPU=True):

    # Parameters
    # **********
    #

    radius = 0.5
    num_per_class = 16384*2
    label_names = {0: 'Unclassified',
                   1: 'Ground',
                   2: 'Building',
                   3: 'Vegetation',
                   4: 'Barriers',
                   5: 'Cars',
                   6: 'Signage'}

    # Collect training features / labels
    # **********************************
    #

    print('Collect Training Features')
    t0 = time.time()
    training_features, training_labels = advanced_point_choice(num_per_class, radius, label_names)
    training_labels -= 1
    training_labels_one_hot = np_utils.to_categorical(training_labels, 6)
    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))


    # Load cloud as a [N x 3] matrix
    cloud_path = '../data/Lille_street_test.ply'
    cloud_ply = read_ply(cloud_path)
    points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
    # Compute features only one time and save them for further use
    #
    #   WARNING : This will save you some time but do not forget to delete your features file if you change
    #             your features. Otherwise you will not compute them and use the previous ones
    #
    feature_file = '../data/Lille_street_test_features_advanced.npy'
    if exists(feature_file):
        features = np.load(feature_file)
    else:
        print('compute feature for test data...')
        t0 = time.time()
        features = compute_features_advanced(points, points, radius)
        np.save(feature_file, features)
        t1 = time.time()
        print('Done in %.3fs\n' % (t1 - t0))

    # Train a MLP
    # ********************************
    #
    device = '/device:GPU:0' if useGPU else '/device:CPU:0'
    with tf.device(device):
        model = Sequential()
        model.add(Dense(200, activation='relu', input_shape=(28,)))
        model.add(Dense(80, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(6, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )




    print('Training MLP')
    t0 = time.time()
    history = model.fit(training_features, training_labels_one_hot,
                        epochs=150, batch_size=2048, verbose=1, validation_split=0.1)
    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))

    # Test
    # ****
    #

    print('Compute testing features')
    t0 = time.time()





    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))

    print('Test')
    t0 = time.time()
    y_prob = model.predict(features)
    predictions = y_prob.argmax(axis=-1)
    predictions += 1
    predictions = predictions.astype(float)
    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))

    print('Save predictions')
    t0 = time.time()
    np.savetxt('../data/Lille_street_test_preds.txt', predictions, fmt='%d')
    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))

    print('Save predictions ply')
    t0 = time.time()
    write_ply('../Lille_street_test_preds.ply',
              [points, predictions],
              ['x', 'y', 'z', 'labels'])
    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))



# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':
    deep_training_and_test()