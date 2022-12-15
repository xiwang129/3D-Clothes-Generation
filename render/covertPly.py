import numpy as np
import open3d as o3d


data = np.load('outfile1.npy')
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(data[0][2])
o3d.io.write_point_cloud('plydata_new_ft.ply',pc)

