
import numpy as np
import math
import sys
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from utilities import normalize_point_cloud
from collections import namedtuple
from utilities import plot_pcd_three_views_color
from tqdm import tqdm
from pprint import pprint
from generator import Generator
from tempfile import TemporaryFile

# modification from the original authur https://github.com/liruihui/SP-GAN 
# contribution includes simplify files, adding conditional labels to generate result, saving point clouds for further rendering

def pc_normalize(pc,return_len=False):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    if return_len:
        return m
    return pc

class Model(object):
    def __init__(self, opts):
        self.opts = opts

    def build_model_eval(self):
        """ Models """
        self.G = Generator(self.opts)
        self.ball = None
  
        print('# generator parameters:', sum(param.numel() for param in self.G.parameters()))

        # self.G.cuda()
        self.G
        self.G.eval()
    
    def read_ball(self,sort=False):
        x = np.loadtxt("2048.xyz")
        ball = pc_normalize(x)

        N = ball.shape[0]
        xx = np.sum(x ** 2, axis=(1)).reshape(N, 1)
        yy = xx.T
        xy = -2 * xx @ yy  # torch.bmm(x, y.permute(0, 2, 1))
        dist = xy + xx + yy  # [B, N, N]
        return ball

    
    def draw_correspondense(self):

        ball = self.read_ball()

        x = np.expand_dims(ball, axis=0)
        ball = np.expand_dims(ball,axis=0)

        self.build_model_eval()

        cat = str(self.opts.choice).lower()
        could_load, save_epoch = self.load(self.opts.log_dir)
        if could_load:
            start_epoch = save_epoch
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            exit(0)
           

        # loop for epoch
        start_time = time.time()
        self.G.eval()

        print(cat, "Start")


        print('# parameters:', sum(param.numel() for param in self.G.parameters()))

        sample_num = 5


        number = 5
        x = np.tile(x, (number, 1, 1))
        x = Variable(torch.Tensor(x))

        for i in range(sample_num):

            noise = np.random.normal(0, 0.2, (number, self.opts.nz))

            color = np.zeros((number+1, self.opts.np, 3))
            color_c = np.squeeze(ball,axis=0)
            color_c = np.minimum(color_c,1.0)
            color_c = np.maximum(color_c,-1.0)

            for j in range(color.shape[0]):
                color[j] = color_c
            title = ["Raw"]
            title.extend(["Sample_%d"% num for num in range(number)])

            noise = np.expand_dims(noise,axis=1)
            noise = np.tile(noise, (1, self.opts.np, 1))

            labels = np.array([0,1])
            labels = np.tile(labels,(sample_num,self.opts.np,4))
            l = torch.from_numpy(labels)

            with torch.no_grad():
                z = Variable(torch.Tensor(noise))
                z_l =  torch.cat([z,l],2)
                out_pc = self.G(x.double(), z_l.double())
                out_pc = out_pc.transpose(2,1)
               

            sample_pcs = out_pc.cpu().detach().numpy()
            sample_pcs = normalize_point_cloud(sample_pcs)

            pcds = np.concatenate([0.5*ball,0.75*sample_pcs],axis=0)

            save_pcds = []
            save_pcds.append(pcds)
            save_pcds = np.array(save_pcds)
            np.save('outfile1',save_pcds)

            plot_pcd_three_views_color("plot-50-new-ft-plot",pcds,title,colors=color)

        del self.G
    

    def load(self, checkpoint_dir):
        if self.opts.pretrain_model_G is None  and self.opts.pretrain_model_D is None:
            print('################ new training ################')
            return False, 1

        print(" [*] Reading checkpoints...")
        
        # ----------------- load G -------------------
        if not self.opts.pretrain_model_G is None:
            resume_file_G = os.path.join(checkpoint_dir, self.opts.pretrain_model_G)
            flag_G = os.path.isfile(resume_file_G), 
            if flag_G == False:
                print('G--> Error: no checkpoint directory found!')
                exit()
            else:
                print('resume_file_G------>: {}'.format(resume_file_G))
                checkpoint = torch.load(resume_file_G,map_location=torch.device('cpu'))
                self.G.load_state_dict(checkpoint['G_model'])
                #self.optimizerG.load_state_dict(checkpoint['G_optimizer'])
                G_epoch = checkpoint['G_epoch']
        else:
            print(" [*] Failed to find the pretrain_model_G")
            exit()

        # ----------------- load D -------------------

        print(" [*] Failed to find the pretrain_model_D")
        #exit()

        print(" [*] Success to load model --> {} & {}".format(self.opts.pretrain_model_G, self.opts.pretrain_model_D))
        return True, G_epoch
        