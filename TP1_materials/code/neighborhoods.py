#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
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
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time

import heapq


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):

    sqr = radius*radius
    neighborhoods = []
    for q in queries:
        d = supports - q
        dists = (d*d).sum(1)
        neighborhoods.append(supports[dists<sqr])


    return neighborhoods


def brute_force_KNN(queries, supports, k):

    neighborhoods = []
    for q in queries:
        p_list = [(1e100, -1)]*k
        d = supports - q
        dists = (d * d).sum(1)
        for i in range(dists.shape[0]):
            if dists[i] < -p_list[0][0]:
                heapq.heapreplace(p_list, (-dists[i], i))


        neighborhood = []
        while p_list:
            neighborhood.append(supports[heapq.heappop(p_list)[1]])

        neighborhoods.append(np.array(neighborhood))

    return neighborhoods


class neighborhood_grid():

    def __init__(self, points, voxel_size):

        #
        #   Tips :
        #       > "__init__" method is called when you create an object of this class with the line :
        #         grid = neighborhood_grid(points, voxel_size)
        #       > You need to keep here the variables that you want to use later (in the query_radius method).
        #         Just call them "self.XXX". The "__init__" method does not need to return anything
        #

        # Example : save voxel size for later use
        self.grid_voxel_size = voxel_size

        self.dict_cell = {}
        default_cell = []
        for p in points:
            i = np.floor(p / voxel_size).astype(int)
            ijk = (i[0], i[1], i[2])
            cell = self.dict_cell.get(ijk, [])
            cell.append(p)
            self.dict_cell[ijk] = cell
        for k, v in self.dict_cell.items():
            self.dict_cell[k] = np.asarray(self.dict_cell[k])

    def query_radius(self, queries, radius):

        #
        #   Tips :
        #       > To speed up the query, you need to find for each point, the part of the grid where its
        #         neighbors can be.
        #       > Then loop over the cells in this part of the grid to find the real neighbors
        #

        # YOUR CODE
        neighborhoods = []
        for q in queries:
            q_max = q+radius
            q_min = q-radius
            ijk_max = np.floor(q_max/self.grid_voxel_size).astype(int)
            ijk_min = np.floor(q_min/self.grid_voxel_size).astype(int)
            ijk_max = (ijk_max[0],ijk_max[1],ijk_max[2])
            ijk_min = (ijk_min[0],ijk_min[1],ijk_min[2])

            neighborhood = []
            for i in range(ijk_min[0], ijk_max[0]+1):
                for j in range(ijk_min[1], ijk_max[1]+1):
                    for k in range(ijk_min[2], ijk_max[2]+1):
                        if (i,j,k) in self.dict_cell:
                            points = self.dict_cell[(i,j,k)]
                            nbs = brute_force_spherical([q], points, radius)
                            nbs = nbs[0].tolist()
                            neighborhood = neighborhood + nbs
            neighborhoods.append(np.asarray(neighborhood))

        return neighborhoods

class oct_cell:
    cells = [(i0, i1, i2) for i0 in [True, False] for i1 in [True, False] for i2 in [True, False]]

    def __init__(self, cloud, center, size, leaf_size=10):
        self.center = center
        self.size = size
        self.cloud = cloud

        if cloud.shape[0] <= leaf_size:
            self.is_leaf = True
            return

        self.is_leaf = False
        self.children = {}
        ind0 = cloud[:, 0] <= center[0]
        ind1 = cloud[:, 1] <= center[1]
        ind2 = cloud[:, 2] <= center[2]
        for i0, i1, i2 in oct_cell.cells:
            part0 = np.logical_xor(not i0, ind0)
            part1 = np.logical_xor(not i1, ind1)
            part2 = np.logical_xor(not i2, ind2)
            part = np.logical_and(part0, np.logical_and(part1, part2))
            child_center = center - np.array([(i0*2-1)*size/2, (i1*2-1)*size/2, (i2*2-1)*size/2])
            self.children[(i0, i1, i2)] = oct_cell(cloud[part], child_center, size/2, leaf_size)

    def query_radius(self, query, radius, points):
        self_radius = np.sqrt(3*self.size*self.size)
        vec = query - self.center
        distance = np.linalg.norm(vec)
        if self_radius + radius < distance:
            return

        if self.is_leaf:
            if self.cloud.shape[0] > 0:
                nbs = brute_force_spherical([query], self.cloud, radius)
                points.extend(list(nbs[0]))
        else:
            if distance + self_radius < radius:
                points.extend(self.cloud)
            else:
                for i0, i1, i2 in oct_cell.cells:
                    self.children[(i0, i1, i2)].query_radius(query, radius, points)



class octree:
    def __init__(self, cloud, leaf_size=10):
        max_coord = np.max(cloud, 0)
        min_coord = np.min(cloud, 0)
        length = np.max(max_coord-min_coord) * 1.1
        center = (max_coord+min_coord)/2

        self.root = oct_cell(cloud, center, length/2, leaf_size)

    def query_radius(self, queries, radius):
        results = []
        for q in queries:
            points=[]
            self.root.query_radius(q, radius, points)
            results.append(np.array(points))
        return results

def verify_neighborhoods(neighborhoods1, neighborhoods2):
    # Compare all neighborhoods
    print('\nVerification of grid neighborhoods :')
    for n1, n2 in zip(neighborhoods1, neighborhoods2):
        if n1.shape[0] != n2.shape[0]:
            print('ERROR FOUND : wrong amount of neighbors')
        else:
            diffs = np.unique(n1, axis=0) - np.unique(n2, axis=0)
            error = np.sum(np.abs(diffs))
            if error > 0:
                print('ERROR FOUND : wrong neighbors')
            else:
                print('This neighborhood is good')

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

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if False:

        # Define the search parameters
        neighbors_num = 1000
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        # Search KNN      
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))

    # Grid neighborhood verification
    # ******************************
    #

    # If statement to skip this part if wanted
    if False:

        # Define the search parameters
        num_queries = 10
        radius = 0.2
        voxel_size = 0.2

        # Create grid structure
        grid = neighborhood_grid(points, voxel_size)

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Get neighborhoods with the grid
        grid_neighborhoods = grid.query_radius(queries, radius)

        # Get neighborhoods with brute force
        brute_neighborhoods = brute_force_spherical(queries, points, radius)

        # Compare all neighborhoods
        verify_neighborhoods(grid_neighborhoods, brute_neighborhoods)

    # Grid neighborhood timings
    # *************************
    #

    # If statement to skip this part if wanted
    if False:

        # Define the search parameters
        num_queries = 10
        radius_values = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
        voxel_size = 1

        # Create grid structure
        grid = neighborhood_grid(points, voxel_size)

        for radius in radius_values:

            # Pick random queries
            random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
            queries = points[random_indices, :]

            # Search spherical
            t0 = time.time()
            if False:
                neighborhoods = grid.query_radius(queries, radius)
            else:
                neighborhoods = brute_force_spherical(queries, points, radius)
            t1 = time.time()        
            print('{:d} spherical neighborhood computed in {:.3f} seconds'.format(num_queries, t1 - t0))

    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if False:
        leaf_sizes = [128]
        for l in leaf_sizes:
            print('leaf_sizes = {:f}'.format(l))
            tree = KDTree(points, leaf_size=l)
            # Define the search parameters
            num_queries = 1000
            radius_values = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
            for radius in radius_values:
                # Pick random queries
                random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
                queries = points[random_indices, :]
                t0 = time.time()
                ind = tree.query_radius(queries, r=radius)
                t1 = time.time()
                print('{:d} spherical neighborhood computed in {:.3f} seconds'.format(num_queries, t1 - t0))

                total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
                print('Computing spherical neighborhoods on whole cloud : {:.0f} seconds'.format(total_spherical_time))

    if True:
        leaf_sizes = [10]
        for l in leaf_sizes:
            print('leaf_sizes = {:f}'.format(l))
            t0 = time.time()
            tree = octree(points, leaf_size=l)
            t1 = time.time()
            print('build octree in {:.3f} seconds'.format(t1 - t0))
            # Define the search parameters
            num_queries = 10
            radius_values = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
            for radius in radius_values:
                # Pick random queries
                random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
                queries = points[random_indices, :]
                t0 = time.time()
                octree_neighborhoods = tree.query_radius(queries, radius)
                t1 = time.time()
                print('{:d} spherical neighborhood computed in {:.3f} seconds'.format(num_queries, t1 - t0))

                # total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
                # print('Computing spherical neighborhoods on whole cloud : {:.0f} seconds'.format(total_spherical_time))
                #
                # Get neighborhoods with brute force
                # brute_neighborhoods = brute_force_spherical(queries, points, radius)
                #
                # Compare all neighborhoods
                # verify_neighborhoods(octree_neighborhoods, brute_neighborhoods)