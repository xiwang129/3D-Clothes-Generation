from plyfile import PlyData
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob 
from utilities import pc_normalize,normalize_point_cloud, toTensorPointCloud,scalePointCloud,gaussianNoisePointCloud
import os

def load_data(filename,num_points =2048):
    f = PlyData.read(filename)
    all_info = f['vertex'][:num_points]
    data = np.array([tuple(i) for i in all_info],dtype = object)
    result = np.array([tuple(i[:3]) for i in data])
    return pc_normalize(result)
 

transformation = transforms.Compose([
    toTensorPointCloud(),
    scalePointCloud(),
    gaussianNoisePointCloud()
])

class plyDataloader(Dataset):
  
    def __init__(self,opts,augment=False):
        self.opts = opts
        self.num_points = opts.np
        self.con = opts.con
  
        pcs = []
        labels = []
        # cats = ["tops","pants","dress"]
        cats = ["tops","dress"]
        
        path = 'dataset/new_point_cloud'
        # ply_file = glob.glob(path + '/*.ply')
        # ply_file = glob.glob(path + '/**/*.ply')
        for i in range(len(cats)):            
            ply_file = glob.glob(os.path.join(path + '/' + cats[i]) + '/*.ply')
            print(ply_file)
            for j in range(len(ply_file)):
                print("---------------ply_file:", ply_file[j])
                pc = load_data(ply_file[j])
                pc = normalize_point_cloud(pc)
                label = np.ones((pc.shape[0],))*i
                pcs.append(pc)
                labels.append(label)
 
        self.labels = np.asarray(labels)
        self.data = np.asarray(pcs)
        self.augment = augment

           
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        point_set = self.data[idx].copy()
        np.random.shuffle(point_set)

        if self.augment:
          point_set = transformation(point_set)
       
        point_set = point_set.astype(np.float32)
        label = self.labels[idx].copy()
        return torch.Tensor(point_set), torch.Tensor(label)
     