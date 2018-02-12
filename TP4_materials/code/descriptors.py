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

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#

def local_PCA(points):
    c = np.mean(points, 0)
    pc = points - c
    N = points.shape[0]
    cov = pc.T.dot(pc)/N

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    return eigenvalues, eigenvectors


def neighborhood_PCA(query_points, cloud_points, radius):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    print("building kdtree..")
    tree = KDTree(cloud_points)
    print("querying ..")
    nbs = tree.query_radius(query_points, radius)
    print("calculating PCA ..")

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))

    for i in range(query_points.shape[0]):
        nb = nbs[i]
        if len(nb) > 0:
            ev, evec = local_PCA(cloud_points[nb,:])
        else:
            ev = np.array([1, 1, 1])
            evec = np.eye(3)
        all_eigenvalues[i, :] = ev
        all_eigenvectors[i, :, :] = evec

    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius):

    # Compute the features for all query points in the cloud
    all_eigenvalues, all_eigenvectors = neighborhood_PCA(query_points, cloud_points, radius)

    l1 = all_eigenvalues[:, 2]
    l2 = all_eigenvalues[:, 1]
    l3 = all_eigenvalues[:, 0]

    normals = all_eigenvectors[:, :, 0]
    ez = np.array([0., 0., 1.])
    verticality = 2 * np.arcsin(np.abs(normals.dot(ez.T))) / np.pi
    l1 += 1e-10
    linearity = 1 - l2/l1
    planarity = (l2-l3)/l1
    sphericity = l3/l1

    return verticality, linearity, planarity, sphericity

def compute_features_with_eig_vec(query_points, cloud_points, radius):

    # Compute the features for all query points in the cloud
    all_eigenvalues, all_eigenvectors = neighborhood_PCA(query_points, cloud_points, radius)

    l1 = all_eigenvalues[:, 2]
    l2 = all_eigenvalues[:, 1]
    l3 = all_eigenvalues[:, 0]

    normals = all_eigenvectors[:, :, 0]
    direction1 = all_eigenvectors[:, :, 2]
    direction2 = all_eigenvectors[:, :, 1]
    ez = np.array([0., 0., 1.])
    verticality = 2 * np.arcsin(np.abs(normals.dot(ez.T))) / np.pi
    l1 += 1e-10
    linearity = 1 - l2/l1
    planarity = (l2-l3)/l1
    sphericity = l3/l1

    return verticality, linearity, planarity, sphericity, normals, direction1, direction2

def compute_features_advanced(query_points, cloud_points, radius):
    verticality, linearity, planarity, sphericity, normals, direction1, direction2 = \
        compute_features_with_eig_vec(query_points, cloud_points, radius)

    features = np.vstack((verticality, linearity, planarity, sphericity))

    for dir in [direction1, -direction1, direction2, -direction2, normals, -normals]:
        verticality, linearity, planarity, sphericity = \
            compute_features(query_points + dir*radius, cloud_points, radius)
        features = np.vstack((features, verticality, linearity, planarity, sphericity))

    return features.T


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = local_PCA(cloud)

        # Print your result
        print(eigenvalues)
        print(eigenvectors)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050576 21.78933008 89.58925317]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

    # Normal computation
    # ******************
    #

    if False:
        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # YOUR CODE
        all_eigenvalues, all_eigenvectors = neighborhood_PCA(cloud, cloud, 0.5)
        normals = all_eigenvectors[:, :, 0]
        write_ply('../Lille_street_small_normals.ply', [cloud, normals], ['x', 'y', 'z', 'nx', 'ny', 'nz'])

    # Features computation
    # ********************
    #

    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        verticality, linearity, planarity, sphericity = compute_features(cloud, cloud, 0.5)
        write_ply('../Lille_street_small_features.ply', [cloud, verticality, linearity, planarity, sphericity], ['x', 'y', 'z', 'vert', 'line', 'plan', 'sph'])

        # YOUR CODE