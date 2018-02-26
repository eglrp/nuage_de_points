#
#
#      0===========================================================0
#      |                      TP6 Modelisation                     |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Plane detection by RANSAC
#
# ------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time

from collections import deque


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
    cov = pc.T.dot(pc) / N

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    return eigenvalues, eigenvectors


def neighborhood_PCA(query_points, cloud_points, tree, radius):
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    print("querying ..")
    t0 = time.time()
    nbs = tree.query_radius(query_points, radius)
    t1 = time.time()
    print("done in {} seconds".format(t1-t0))
    print("calculating PCA ..")

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))

    for i in range(query_points.shape[0]):
        nb = nbs[i]
        if len(nb) > 0:
            ev, evec = local_PCA(cloud_points[nb, :])
        else:
            ev = np.array([1, 1, 1])
            evec = np.eye(3)
        all_eigenvalues[i, :] = ev
        all_eigenvectors[i, :, :] = evec

    return all_eigenvalues, all_eigenvectors


def compute_curvatures_and_normals(points, search_tree, radius):
    all_eigenvalues, all_eigenvectors = neighborhood_PCA(points, points, search_tree, radius)

    curvatures = all_eigenvalues[:, 0] / np.sum(all_eigenvalues, 1)
    normals = all_eigenvectors[:, :, 0]
    return curvatures, normals


def region_criterion(p1, p2, n1, n2):
    inplane = np.abs(np.dot(p1 - p2, n1))
    align = 1.0 - np.abs(np.dot(n1, n2))
    return inplane < 1e-2 and align < 5e-5

def RegionGrowing(cloud, normals, curvatures, search_tree, radius, region_criterion):
    region = np.zeros(len(cloud), dtype=bool)

    seed = np.random.randint(cloud.shape[0])
    Q = deque()
    Q.append(seed)
    while len(Q) > 0:
        q = Q.popleft()
        nbs = search_tree.query_radius([cloud[q]], radius)
        for p in nbs[0]:
            if region[p]:
                continue
            if region_criterion(cloud[q], cloud[p], normals[q], normals[p]):
                region[p] = True
                if curvatures[p] < 1e-2:
                    Q.append(p)
    return region


def recursive_RegionGrowing(cloud, normals, curvatures, radius, region_criterion,
                            NB_PLANES=2):
    plane_inds = np.array([], dtype=int)
    remaining_inds = np.arange(cloud.shape[0], dtype=int)
    plane_labels = np.array([-1] * cloud.shape[0], dtype=int)
    for i in range(NB_PLANES):
        search_tree = KDTree(cloud[remaining_inds], leaf_size=100)
        region = RegionGrowing(cloud[remaining_inds], normals[remaining_inds], curvatures[remaining_inds],
                               search_tree, radius, region_criterion)

        good_idx = remaining_inds[region]
        plane_inds = np.union1d(plane_inds, good_idx)
        plane_labels[good_idx] = i
        remaining_inds = np.setdiff1d(remaining_inds, plane_inds)

    return plane_inds, remaining_inds, plane_labels[plane_labels != -1]


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan_sub2cm.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']
    N = len(points)

    # Computes Normals of the whole cloud
    # ************************
    #
    if True:
        print('\n--- Question 5 ---\n')

        # Build a search tree
        t0 = time.time()
        search_tree = KDTree(points, leaf_size=10)
        t1 = time.time()
        print('KDTree computation done in {:.3f} seconds'.format(t1 - t0))

        # Parameters for normals computation
        radius = 0.3

        # Computes normals of the whole cloud
        t0 = time.time()
        curvatures, normals = compute_curvatures_and_normals(points, search_tree, radius)
        t1 = time.time()
        print('normals and curvatures computation done in {:.3f} seconds'.format(t1 - t0))

        # Save
        # write_ply('../decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])


        # Find a plane by Region Growing
        # ***********************************
        #
        #
    if False:
        print('\n--- Questions 6 and 7 ---\n')

        # Define parameters of Region Growing
        radius = 0.5

        # Find a plane by Region Growing
        t0 = time.time()
        region = RegionGrowing(points, normals, curvatures, search_tree, radius, region_criterion)
        t1 = time.time()
        print('Region Growing done in {:.3f} seconds'.format(t1 - t0))

        #
        plane_inds = region.nonzero()[0]
        remaining_inds = (1 - region).nonzero()[0]

        # Save the best plane
        write_ply('../best_plane.ply',
                  [points[plane_inds], colors[plane_inds], labels[plane_inds], curvatures[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'curvatures'])
        write_ply('../remaining_points.ply',
                  [points[remaining_inds], colors[remaining_inds], labels[remaining_inds], curvatures[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'curvatures'])

    # Find "all planes" in the cloud
    # ***********************************
    #
    #
    if True:
        print('\n--- Question 7 ---\n')

        radius = 0.1
        NB_PLANES = 20

        # Recursively find best plane
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = recursive_RegionGrowing(points, normals, curvatures,
                                                                           radius, region_criterion,
                                                                           NB_PLANES)
        t1 = time.time()
        print('recursive RegionGrowing done in {:.3f} seconds'.format(t1 - t0))

        # Save the best planes and remaining points
        write_ply('../best_planes.ply',
                  [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
        write_ply('../remaining_points_.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

        print('Done')
