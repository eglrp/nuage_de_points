from plyfile import PlyData, PlyElement
import numpy as np

# p = PlyData.read("../data/feature_1.ply")
# print(p)

a = np.array([(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)],
             dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
print(a)
el = PlyElement.describe(a, 'myvertices')
PlyData([el]).write('some_binary.ply')
PlyData([el], text=True).write('some_ascii.ply')
