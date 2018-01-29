#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
#


#------------------------------------------------------------------------------------------
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
from utils.visu import show_ICP


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''
    d = data.shape[0]
    bc_data = np.mean(data, 1).reshape(d, 1)
    bc_ref = np.mean(ref, 1).reshape(d, 1)
    c_data = data - bc_data
    c_ref = ref - bc_ref
    H = c_data.dot(c_ref.T)
    u, s, v = np.linalg.svd(H)
    v = v.T

    # YOUR CODE
    R = v.dot(u.T)
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = v.dot(u.T)
    T = bc_ref - R.dot(bc_data)

    return R, T


def icp_point_to_point(data, ref, max_iter, RMS_threshold, number_limit=0):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    tree = KDTree(ref.T, leaf_size=10)
    n = data.shape[1]

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []

    for i in range(max_iter):

        if number_limit>0:
            indices = np.random.choice(n, number_limit, replace=False)
            data_aligned_chosen = data_aligned[:, indices]
            data_chosen = data[:, indices]
        else:
            data_aligned_chosen = data_aligned
            data_chosen = data

        ids = tree.query(data_aligned_chosen.T, k=1, return_distance=False).flatten()
        ref_points = ref[:,ids]
        R, T = best_rigid_transform(data_chosen, ref_points)
        data_aligned = R.dot(data) + T


        print(i, RMS(data_aligned_chosen, ref_points))

        R_list.append(R)
        T_list.append(T)
        neighbors_list.append(ids)

    return data_aligned, R_list, T_list, neighbors_list

def RMS(cloud1, cloud2):
    diff = cloud1 - cloud2
    return np.sqrt(np.sum(diff**2) / diff.shape[1])

#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':
   
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'

        # Load clouds
        bunny_o = read_ply(bunny_o_path)
        bunny_o = np.vstack((bunny_o['x'], bunny_o['y'], bunny_o['z']))
        bunny_r = read_ply(bunny_r_path)
        bunny_r = np.vstack((bunny_r['x'], bunny_r['y'], bunny_r['z']))
        # Find the best transformation
        R, T = best_rigid_transform(bunny_r, bunny_o)
        # Apply the tranformation
        bunny_t = R.dot(bunny_r)+T

        # Save cloud
        write_ply('../bunny_transformed.ply', [bunny_t.T], ['x', 'y', 'z'])

        # Compute RMS
        diff = bunny_t - bunny_o
        rms = RMS(bunny_t, bunny_o)
        # Print RMS
        print("RMS = ", RMS)

    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'

        # Load clouds
        ref2D = read_ply(ref2D_path)
        ref2D = np.vstack((ref2D['x'], ref2D['y']))
        data2D = read_ply(data2D_path)
        data2D = np.vstack((data2D['x'], data2D['y']))
        # Apply ICP
        data_aligned, R_list, T_list, neighbors_list = icp_point_to_point(data2D, ref2D, 50, 1e-6)
        # Show ICP
        show_ICP(data2D, ref2D, R_list, T_list, neighbors_list)

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'

        # Load clouds
        bunny_o = read_ply(bunny_o_path)
        bunny_o = np.vstack((bunny_o['x'], bunny_o['y'], bunny_o['z']))
        bunny_r = read_ply(bunny_p_path)
        bunny_r = np.vstack((bunny_r['x'], bunny_r['y'], bunny_r['z']))
        # Apply ICP
        data_aligned, R_list, T_list, neighbors_list = icp_point_to_point(bunny_r, bunny_o, 50, 1e-6)
        # Show ICP
        show_ICP(bunny_o, bunny_r, R_list, T_list, neighbors_list)


    # Fast ICP
    # ********
    #

    # If statement to skip this part if wanted
    if True:

        # Cloud paths
        NDDC_1_path = '../data/Notre_Dame_Des_Champs_1.ply'
        NDDC_2_path = '../data/Notre_Dame_Des_Champs_2.ply'

        # Load clouds
        NDDC_1 = read_ply(NDDC_1_path)
        NDDC_1 = np.vstack((NDDC_1['x'], NDDC_1['y'], NDDC_1['z']))
        NDDC_2 = read_ply(NDDC_2_path)
        NDDC_2 = np.vstack((NDDC_2['x'], NDDC_2['y'], NDDC_2['z']))
        # Apply ICP
        # data_aligned, R_list, T_list, neighbors_list = icp_point_to_point(NDDC_2, NDDC_1, 50, 1e-6)

        # Apply fast ICP for different values of the sampling_limit parameter
        data_aligned, R_list, T_list, neighbors_list = icp_point_to_point(NDDC_2, NDDC_1, 50, 1e-6, number_limit=500000)

