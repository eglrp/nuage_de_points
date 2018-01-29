#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      First script of the practical session. Transformation of a point cloud
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

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#
def center_points(cloud):
    mean = np.mean(cloud, 0)
    return cloud - mean


def rotation_z(cloud, theta):
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    cloud[:, 0:2] = np.dot(cloud[:, 0:2], rot.T)
    return cloud

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

    # Path of the file
    file_path = '../data/bunny.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate x, y, and z in a (N*3) point matrix
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Concatenate R, G, and B channels in a (N*3) color matrix
    colors = np.vstack((data['red'], data['green'], data['blue'])).T

    # Get the scalar field which represent density as a vector
    density = data['scalar_density']

    # Transform point cloud
    # *********************
    #
    #   Follow the instructions step by step
    #

    # Replace this line by your code
    transformed_points = center_points(points)
    transformed_points = transformed_points / 2.
    transformed_points = rotation_z(transformed_points, -np.pi/2.)
    transformed_points = center_points(transformed_points)
    transformed_points[:, 1] -= 0.10
    # Save point cloud
    # *********************
    #
    #   Save your result file
    #   (See write_ply function)
    #

    # Save point cloud
    write_ply('../little_bunny.ply', [transformed_points, colors, density],
              ['x', 'y', 'z', 'red', 'green', 'blue', 'density'])

    print('Done')
