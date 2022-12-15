import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities import get_edge_features
#from torchsummary import summary

neg = 0.01
neg2 = 0.2
num_classes = 2

# modification made from original author https://github.com/liruihui/SP-GAN 
# contribution including ensure the pretrained generator model loaded correctly and change the input dimentions to accommodate class labels 

class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channel, style_dim, use_eql=False):
        super().__init__()
        Conv = nn.Conv1d
        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = Conv(style_dim,in_channel*2,1).to(torch.double)

        self.style.weight.data.normal_()
        self.style.bias.data.zero_()

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0
    
    def forward(self,input,style):
        style = self.style(style)
        gamma,beta = style.chunk(2,1)

        out = self.norm(input)
        out = gamma * out + beta

        return out

class EdgeBlock(nn.Module):
    def __init__ (self,Fin,Fout,k, attn=True):
        super(EdgeBlock,self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.conv_w = nn.Sequential(
            nn.Conv2d(Fin,Fout//2,1).to(torch.double),
            nn.BatchNorm2d(Fout//2).to(torch.double),
            nn.LeakyReLU(neg,inplace=True),
            nn.Conv2d(Fout //2, Fout,1).to(torch.double),
            nn.BatchNorm2d(Fout).to(torch.double),
            nn.LeakyReLU(neg,inplace=True),
        )

        self.conv_x = nn.Sequential(
            nn.Conv2d(2*Fin, Fout,[1,1],[1,1]).to(torch.double),
            nn.BatchNorm2d(Fout).to(torch.double),
            nn.LeakyReLU(neg,inplace=True)
        )

        self.conv_out = nn.Conv2d(Fout,Fout,[1,k],[1,1]).to(torch.double)

    def forward(self,x):
        B,C,N = x.shape
        x = get_edge_features(x,self.k)
        w = self.conv_w(x[:,C:,:,:])
        w = F.softmax(w,dim=-1)

        x = self.conv_x(x)
        x = x*w
        x = self.conv_out(x)
        x = x.squeeze(3)

        return x


class Generator(nn.Module):
    def __init__(self,opts):
        super(Generator,self).__init__()
        self.opts = opts
        self.np = opts.np
        self.nk = opts.nk // 2
        self.nz = opts.nz
        softmax = opts.softmax
        self.off = opts.off 
        self.use_attn = opts.attn
        self.use_head = opts.use_head

        Conv = nn.Conv1d
        Linear = nn.Linear

        dim =128

        if self.opts.pretrain_model_G:
            self.head = nn.Sequential(
                Conv(3+self.nz,dim,1).to(torch.double),
                nn.LeakyReLU(neg,inplace=True),
                Conv(dim,dim,1).to(torch.double),
                nn.LeakyReLU(neg,inplace=True),
            )
        else:
            self.head = nn.Sequential(
                Conv(3+self.nz+num_classes*4,dim,1).to(torch.double),
                nn.LeakyReLU(neg,inplace=True),
                Conv(dim,dim,1).to(torch.double),
                nn.LeakyReLU(neg,inplace=True),
            )

        self.global_conv = nn.Sequential(
            Linear(dim,dim).to(torch.double),
            nn.BatchNorm1d(dim).to(torch.double),
            nn.LeakyReLU(neg,inplace=True),
            Linear(dim,512).to(torch.double),
            nn.BatchNorm1d(512).to(torch.double),
            nn.LeakyReLU(neg,inplace=True),
        )

        self.tail = nn.Sequential(
            nn.Conv1d(512 +dim,256,1).to(torch.double),
            nn.LeakyReLU(neg,inplace=True),
            nn.Conv1d(256,64,1).to(torch.double),
            nn.LeakyReLU(neg,inplace=True),
            nn.Conv1d(64,3,1).to(torch.double),
            nn.Tanh()
        )

        if self.use_head:
            self.pc_head = nn.Sequential(
            Conv(3,dim//2,1).to(torch.double),
            nn.LeakyReLU(inplace=True),
            Conv(dim //2,dim,1).to(torch.double),
            nn.LeakyReLU(inplace=True),
        )
            self.EdgeConv1 = EdgeBlock(dim,dim,self.nk)
            self.adain1 = AdaptivePointNorm(dim,dim)
            self.EdgeConv2 = EdgeBlock(dim,dim,self.nk)
            self.adain2 = AdaptivePointNorm(dim,dim)

        else:
            self.EdgeConv1 = EdgeBlock(3,64,self.nk)
            self.adain1 = AdaptivePointNorm(64,dim)
            self.EdgeConv2 = EdgeBlock(64,dim,self.nk)
            self.adain2 = AdaptivePointNorm(dim,dim)
     
        self.lrelu1 = nn.LeakyReLU(neg2)
        self.lrelu2 = nn.LeakyReLU(neg2)
    
    def forward(self,x,z):

        B,N,_ = x.size()
        if self.opts.z_norm:
            z = z / (z.norm(p=2,dim=-1, keepdim=True)+1e-8)
    
        style = torch.cat([x,z],dim=-1)
        style = style.transpose(2,1).contiguous()
        style = self.head(style)

        
        pc = x.transpose(2,1).contiguous()
        if self.use_head:
            pc = self.pc_head(pc)
    
        x1 = self.EdgeConv1(pc)
        x1 = self.lrelu1(x1)
        x1 = self.adain1(x1, style)

        x2 = self.EdgeConv2(x1)
        x2 = self.lrelu2(x2)
        x2 = self.adain2(x2, style) 

        feat_global = torch.max(x2,2,keepdim= True)[0]
        feat_global = feat_global.view(B,-1)
        feat_global = self.global_conv(feat_global)
        feat_global = feat_global.view(B,-1,1)
        feat_global = feat_global.repeat(1,1,N)

        feat_cat = torch.cat((feat_global,x2),dim=1)

        if self.use_attn:
            feat_cat = self.attn(feat_cat)
        
        x1_o = self.tail(feat_cat)
        x1_p = pc + x1_o if self.off else x1_o

        return x1_p






       
