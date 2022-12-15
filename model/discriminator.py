import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

neg =[1e-2,0.2][0]
num_classes = 2

# original author https://github.com/liruihui/SP-GAN 

class BasicConv1D(nn.Module):
    def __init__(self, Fin, Fout, act=True, norm="BN", kernal=1):
        super(BasicConv1D, self).__init__()

        self.conv = nn.Conv1d(Fin,Fout,kernal)
        if act:
            self.act = nn.LeakyReLU(inplace=True)
        else:
            self.act = None

        if norm is not None:
            self.norm = nn.BatchNorm1d(Fout) if norm=="BN" else nn.InstanceNorm1d(Fout)
        else:
            self.norm = None

    def forward(self, x):
        x = self.conv(x)  # Bx2CxNxk

        if self.norm is not None:
            x = self.norm(x)

        if self.act is not None:
            x = self.act(x)

        return x
        
class Discriminator(nn.Module):
    def __init__(self,opts,num_points =2048):
        super(Discriminator, self).__init__()
        self.num_point = num_points
        BN = True 
        self.small_d = opts.small_d
    
        self.mlps = nn.Sequential(
            nn.Conv1d(3+num_classes,64,1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(neg,inplace=True),
            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(neg,inplace=True),
            nn.Conv1d(128,256,1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(neg,inplace=True),
        )

        self.mode = ['max','max_avg'][0]

        if self.mode == 'max':
            dim = 1024
        else:
            dim =512
        
        if self.small_d:
            dim = dim//2

        
        self.fc2 = nn.Sequential(
            nn.Conv1d(256,dim,1),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg,inplace=True)
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim,512),
            nn.LeakyReLU(neg, inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(neg,inplace=True),
            nn.Linear(256,64),
            nn.LeakyReLU(neg,inplace=True),
            nn.Linear(64,1),
        )
    
    def forward(self,x):
        B = x.size()[0]

        x = self.mlps(x)
        x = self.fc2(x)

        x1 = F.adaptive_max_pool1d(x,1).view(B,-1)

        if self.mode == "max":
            x =x1
        else:
            x2 = F.adaptive_avg_pool1d(x,1).view(B,-1)
            x = torch.cat((x1,x2),1)
        
        x3 = self.mlp(x)

        return x3



