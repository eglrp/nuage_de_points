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

import time
from utils.withtimer import Timer

from multiprocessing import Pool


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#

def local_PCA(points):
    if points.shape[0] == 0:
        ev = np.array([1, 1, 1])
        evec = np.eye(3)
        return ev, evec

    c = np.mean(points, 0)
    pc = points - c
    N = points.shape[0]
    cov = pc.T.dot(pc) / N

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    return eigenvalues, eigenvectors


def neighborhood_PCA(query_points, cloud_points, radius):
    p = Pool()
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    with Timer("build kdtree"):
        tree = KDTree(cloud_points)
    with Timer("querying"):
        nbs = tree.query_radius(query_points, radius)
        # nbs = p.starmap(tree.query_radius, [([pt], radius) for pt in query_points])

    # all_eigenvalues = np.zeros((query_points.shape[0], 3))
    # all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))
    with Timer("calculating PCA"):
        with Timer("dispatch_cloud"):
            dispatch_cloud = [cloud_points[idx] for idx in nbs]
        r = p.map(local_PCA, dispatch_cloud)
        p.close()
        p.join()
        all_eigenvalues, all_eigenvectors = zip(*r)


        # for i in range(query_points.shape[0]):
        #     nb = nbs[i]
        #     ev, evec = local_PCA(cloud_points[nb, :])
        #     all_eigenvalues[i, :] = ev
        #     all_eigenvectors[i, :, :] = evec

    return np.array(all_eigenvalues), np.array(all_eigenvectors)

def compute_features_with_eigen(all_eigenvalues, all_eigenvectors):
    l1 = all_eigenvalues[:, 2]
    l2 = all_eigenvalues[:, 1]
    l3 = all_eigenvalues[:, 0]

    normals = all_eigenvectors[:, :, 0]
    ez = np.array([0., 0., 1.])
    verticality = 2 * np.arcsin(np.abs(normals.dot(ez.T))) / np.pi
    l1 += 1e-10
    linearity = 1 - l2 / l1
    planarity = (l2 - l3) / l1
    sphericity = l3 / l1

    return verticality, linearity, planarity, sphericity


def compute_features_advanced(query_pts, cloud_pts, radius):
    all_ev, all_evec = neighborhood_PCA(query_pts, cloud_pts, radius)

    normals = all_evec[:, :, 0]
    direction1 = all_evec[:, :, 2]
    direction2 = all_evec[:, :, 1]

    feats = compute_features_with_eigen(all_ev, all_evec)

    features = np.vstack(feats)

    for direction in [direction1, -direction1, direction2, -direction2, normals, -normals]:
        all_ev, all_evec = neighborhood_PCA(query_pts + direction * radius, cloud_pts, radius)
        feats = compute_features_with_eigen(all_ev, all_evec)
        features = np.vstack((features,feats))

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
        radius = 0.5

        all_ev, all_evec = neighborhood_PCA(cloud, cloud, radius)
        verticality, linearity, planarity, sphericity = compute_features_with_eigen(all_ev, all_evec)
        write_ply('../Lille_street_small_features.ply', [cloud, verticality, linearity, planarity, sphericity],
                  ['x', 'y', 'z', 'vert', 'line', 'plan', 'sph'])
