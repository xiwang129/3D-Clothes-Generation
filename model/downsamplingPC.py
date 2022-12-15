import numpy as np
from plyfile import PlyData,PlyElement
import glob
import os
from pathlib import Path

def downsampling(filename):        
    data =  PlyData.read(filename)
    all = data['vertex']
  
    if len(all) > 105000:
        factor = 50
        downsample_all = all[::factor]
        all_info = np.array([tuple(i) for i in downsample_all],dtype = [('x','f4'),('y','f4'),('z','f4'),('nx','f4'),('ny','f4'),('nz','f4'),('red','f4'),('green','f4'),('blue','f4'),('alpha','u1')])
    else:
        factor = 10
        downsample_all = all[::factor]
        all_info = np.array([tuple(i) for i in downsample_all],dtype = [('x','f4'),('y','f4'),('z','f4'),('nx','f4'),('ny','f4'),('nz','f4'),('red','f4'),('green','f4'),('blue','f4'),('alpha','u1')])

    el4 = PlyElement.describe(all_info,'vertex')
    path = Path(filename).name
    result = PlyData([el4]).write('new_'+ path)
    return result

  
if __name__ == "__main__":
    path = '/Users/amy/cv-final-project-ft-con/dataset/pants'
    ply_file = glob.glob(path + '/**/*.ply')
    for i in range(len(ply_file)):
        file = ply_file[i]
        pc = downsampling(file)
   