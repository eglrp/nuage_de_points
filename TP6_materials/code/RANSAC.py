#
#
#      0===========================================================0
#      |                      TP6 Modelisation                     |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      First script of the practical session. Plane detection by RANSAC
#
#------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time



#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_plane(points):

    p0 = points[0]
    p1 = points[1]
    p2 = points[2]

    normal = np.cross(p1-p0, p2-p0)
    normal = normal / np.linalg.norm(normal)

    
    return p0, normal


def in_plane(points, ref_pt, normal, threshold_in=0.1):
    d = np.dot(points - ref_pt, normal.T)
    return np.abs(d) < threshold_in

def RANSAN_refine(points, subpoints, NB_RANDOM_REFINES=100, threshold_in=0.1):
    M = subpoints.shape[0]
    best_count = -1
    for i in range(NB_RANDOM_REFINES):
        draw = np.random.choice(M, 3)
        p, n = compute_plane(subpoints[draw])

        is_in = in_plane(points, p, n, threshold_in)
        count = np.sum(is_in)
        if count > best_count:
            best_count = count
            best_ref_pt = p
            best_normal = n

    return best_ref_pt, best_normal


def RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1):
    best_ref_pt = np.zeros((3, 1))
    best_normal = np.zeros((3, 1))
    best_density = -1
    N = points.shape[0]
    corner1 = np.max(points, 0)
    corner2 = np.min(points, 0)
    l = np.abs(corner1-corner2)

    volume = l[0]*l[1]*l[2]
    area0 = volume / l[0]
    area1 = volume / l[1]
    area2 = volume / l[2]

    trials = 0
    for i in range(NB_RANDOM_DRAWS):
        trials += 1
        draw = np.random.choice(N, 3)
        p, n = compute_plane(points[draw])

        intersect_a0 = area0 / abs(n[0])
        intersect_a1 = area1 / abs(n[1])
        intersect_a2 = area2 / abs(n[2])
        intersection_area = np.min(np.array([intersect_a0, intersect_a1, intersect_a2]))
        plane_volume = 2 * threshold_in * intersection_area

        is_in = in_plane(points, p, n, threshold_in)
        density = np.sum(is_in) / plane_volume
        if density > best_density:
            best_density = density
            best_ref_pt = p
            best_normal = n

    #return RANSAN_refine(points, points[is_in], 100, threshold_in)
    return best_ref_pt, best_normal


def recursive_RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1, NB_PLANES=2):
    plane_inds = np.array([], dtype=int)
    remaining_inds = np.arange(points.shape[0], dtype=int)
    plane_labels = np.array([-1]*points.shape[0], dtype=int)
    for i in range(NB_PLANES):
        p, n = RANSAC(points[remaining_inds], NB_RANDOM_DRAWS, threshold_in)
        idx = in_plane(points[remaining_inds], p, n, threshold_in)
        good_idx = remaining_inds[idx]
        plane_inds = np.union1d(plane_inds, good_idx)
        plane_labels[good_idx] = i
        remaining_inds = np.setdiff1d(remaining_inds, plane_inds)
    
    return plane_inds, remaining_inds, plane_labels[plane_labels!=-1]


#------------------------------------------------------------------------------------------
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
    
    if False:
    # Computes the plane passing through 3 randomly chosen points
    # ************************
    #
    
        print('\n--- Question 1 and 2 ---\n')

        # Define parameter
        threshold_in = 0.1

        # Take randomly three points
        pts = points[np.random.randint(0, N, size=3)]

        # Computes the plane passing through the 3 points
        t0 = time.time()
        ref_pt, normal = compute_plane(pts)
        t1 = time.time()
        print('plane computation done in {:.3f} seconds'.format(t1 - t0))

        # Find points in the plane and others
        t0 = time.time()
        points_in_plane = in_plane(points, ref_pt, normal, threshold_in)
        t1 = time.time()
        print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
        plane_inds = points_in_plane.nonzero()[0]
        remaining_inds = (1-points_in_plane).nonzero()[0]

        # Save extracted plane and remaining points
        write_ply('../plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        write_ply('../remaining_points.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])


        # Computes the best plane fitting the point cloud
        # ***********************************
        #
        #

        print('\n--- Question 3 ---\n')

        # Define parameters of RANSAC
        NB_RANDOM_DRAWS = 100
        threshold_in = 0.05

        # Find best plane by RANSAC
        t0 = time.time()
        best_ref_pt, best_normal = RANSAC(points, NB_RANDOM_DRAWS, threshold_in)
        t1 = time.time()
        print('RANSAC done in {:.3f} seconds'.format(t1 - t0))

    #    print(best_ref_pt, best_normal, "\tNb of votes:",best_vote)

        # Find points in the plane and others
        points_in_plane = in_plane(points, ref_pt, normal, threshold_in)
        plane_inds = points_in_plane.nonzero()[0]
        remaining_inds = (1-points_in_plane).nonzero()[0]

        # Save the best extracted plane and remaining points
        write_ply('../best_plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        write_ply('../remaining_points.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])


    # Find "all planes" in the cloud
    # ***********************************
    #
    #
    if True:
        print('\n--- Question 4 ---\n')

        # Define parameters of recursive_RANSAC
        NB_RANDOM_DRAWS = 200
        threshold_in = 0.05
        NB_PLANES = 10

        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points, NB_RANDOM_DRAWS, threshold_in, NB_PLANES)
        t1 = time.time()
        print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Save the best planes and remaining points
        write_ply('../best_planes.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
        write_ply('../remaining_points_.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])



    print('Done')
