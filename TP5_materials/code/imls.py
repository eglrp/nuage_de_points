#
#
#      0===========================================================0
#      |              TP5 Surface Reconstruction                   |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 02/02/2018
#



# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time




if __name__ == '__main__':

    # Path of the file
    file_path = '../data/bunny_normals.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

    tree = KDTree(points, leaf_size=2)

    min_grid = np.min(points, 0)
    max_grid = np.max(points, 0)

    number_cells = 100 #100
    N = number_cells + 1

    scalar_field = np.zeros((N**3, 4))
    tree = KDTree(points)
    count = 0
    xs = np.linspace(min_grid[0], max_grid[0], N)
    ys = np.linspace(min_grid[1], max_grid[1], N)
    zs = np.linspace(min_grid[2], max_grid[2], N)
    X, Y, Z = np.meshgrid(xs, ys, zs)
    grid = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T
    indices = tree.query(grid, 1, return_distance=False)
    for i in range(indices.shape[0]):
        idx = indices[i][0]
        point = points[idx]
        normal = normals[idx]
        fx = normal.dot(grid[i]-point)
        scalar_field[i, :] = np.hstack((point, fx))




    file = open('grid.csv','w')
    file.write('x coord, y coord, z coord, scalar\n')
    for s in scalar_field:
        file.write(str(s[0]) + ', ' + str(s[1]) + ', ' + str(s[2]) + ', ' + str(s[3]) + '\n')

    file.close()



