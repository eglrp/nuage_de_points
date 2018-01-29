#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
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
from sklearn.preprocessing import label_binarize

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
#   Here you can define useful functions to be used in the main
#


def cloud_decimation(points, colors, labels, factor):
    decimated_points = points[::factor]
    decimated_colors = colors[::factor]
    decimated_labels = labels[::factor]

    return decimated_points, decimated_colors, decimated_labels


def grid_subsampling(points, voxel_size):

    #   Tips :
    #       > First compute voxel indices in each direction for all the points (can be negative).
    #       > Sum and count the points in each voxel (use a dictionaries with the indices as key).
    #         Remember arrays cannot be dictionary keys, but tuple can.
    #       > Divide the sum by the number of point in each cell.
    #       > Do not forget you need to return a numpy array and not a dictionary.
    #

    dic_sum = {}
    default_cell = (np.array([0.,0.,0.]), 0)
    for p in points:
        i = np.floor(p/voxel_size).astype(int)
        ijk = (i[0],i[1],i[2])
        cell = dic_sum.get(ijk, default_cell)
        cell = (cell[0]+p, cell[1]+1)
        dic_sum[ijk] = cell
    subsampled_points = []
    for _, value in dic_sum.items():
        subsampled_points.append(value[0]/value[1])

    subsampled_points = np.array(subsampled_points)

    return subsampled_points


def grid_subsampling_colors_labels(points, colors_int, labels, voxel_size):
    colors = colors_int.astype(float)
    label_classes = np.unique(labels)
    labels_binary = label_binarize(labels, classes=label_classes)
    dic_sum = {}
    default_cell = (np.zeros(3), np.zeros(3), np.zeros(len(label_classes)), 0)
    for k in range(points.shape[0]):
        p = points[k]
        c = colors[k]
        l = labels_binary[k]
        i = np.floor(p / voxel_size).astype(int)
        ijk = (i[0], i[1], i[2])
        cell = dic_sum.get(ijk, default_cell)
        cell = (cell[0] + p, cell[1] + c, cell[2] + l, cell[3] + 1)
        dic_sum[ijk] = cell

    subsampled_points = []
    subsampled_colors = []
    subsampled_labels = []
    for _, value in dic_sum.items():
        subsampled_points.append(value[0] / value[3])
        subsampled_colors.append(value[1] / value[3])
        subsampled_labels.append(value[2].argmax())
    subsampled_points = np.array(subsampled_points)
    subsampled_colors = np.array(subsampled_colors).astype('uint8')
    subsampled_labels = np.array(subsampled_labels).astype(int)

    return subsampled_points, subsampled_colors, subsampled_labels


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
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']    

    # Decimate the point cloud
    # ************************
    #

    # Define the decimation factor
    factor = 300

    # Decimate
    t0 = time.time()
    decimated_points, decimated_colors, decimated_labels = cloud_decimation(points, colors, labels, factor)
    t1 = time.time()
    print('decimation done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    # Subsample the point cloud on a grid
    # ***********************************
    #
    if False:
        # Define the size of the grid
        voxel_size = 0.2

        # Subsample
        t0 = time.time()
        subsampled_points = grid_subsampling(points, voxel_size)
        t1 = time.time()
        print('Subsampling done in {:.3f} seconds'.format(t1 - t0))

        # Save
        write_ply('../grid_subsampled.ply', [subsampled_points], ['x', 'y', 'z'])

    # Subsample with color and label the point cloud on a grid
    # ***********************************
    #

    # Define the size of the grid
    voxel_size = 0.2

    # Subsample
    t0 = time.time()
    subsampled_points, subsampled_colors, subsampled_labels = \
        grid_subsampling_colors_labels(points, colors, labels, voxel_size)
    t1 = time.time()
    print('Subsampling done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../grid_subsampled_color.ply',
              [subsampled_points, subsampled_colors, subsampled_labels],
              ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])


    print('Done')
