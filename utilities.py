import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import functools

# original author https://github.com/liruihui/SP-GAN 

class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def smooth_labels(B,ran=[0.9,1.0]):
    #return y - 0.3 + (np.random.random(y.shape) * 0.5)
    return (ran[1]-ran[0])*np.random.random(B) + ran[0]

def noisy_labels(y, p_flip=0.05):
    # determine the number of labels to flip
    n_select = int(p_flip * y.shape[0])
    # choose labels to flip
    flip_ix = np.random.choice([i for i in range(y.shape[0])], size=n_select)
    # invert the labels in place
    y[flip_ix] = 1 - y[flip_ix]
    return y

def BCEloss(D_fake, D_real, d_real_target, d_fake_target):
    real =  F.binary_cross_entropy_with_logits(D_real,d_real_target.expand_as(D_real)).cuda()
    fake =  F.binary_cross_entropy_with_logits(D_fake,d_fake_target.expand_as(D_fake)).cuda()
    return real, fake

def BCEfakeloss(D_fake,target):
    return F.binary_cross_entropy_with_logits(D_fake, target.expand_as(D_fake)).cuda()


def dis_loss(d_real, d_fake, gan="wgan", weight=1.,d_real_p=None, d_fake_p=None, noise_label=False):
    # B = d_fake.size(0)
    # a = 1.0
    # b = 0.9

    if gan.lower() == "wgan":
        loss_fake = d_fake.mean()
        loss_real = d_real.mean()
        wg_loss_orig = loss_fake - loss_real
        wg_loss = wg_loss_orig * weight
        return wg_loss, {
            "wgan_dis_loss": wg_loss.clone().detach().item(),
            "wgan_dis_loss_orig": wg_loss_orig.clone().detach().item(),
            "wgan_dis_loss_real": loss_real.clone().detach().item(),
            "wgan_dis_loss_fake": loss_fake.clone().detach().item()
        }
    elif gan.lower() == "hinge":
        d_loss_real = torch.nn.ReLU()(1.0 - d_real).mean()
        d_loss_fake = torch.nn.ReLU()(1.0 + d_fake).mean()

        # d_loss_real = -torch.min(d_real - 1, d_real * 0).mean()
        # d_loss_fake = -torch.min(-d_fake - 1, d_fake * 0).mean()
        real_correct = (d_real >= 0.).float().sum() + (d_fake < 0.).float().sum()
        real_acc = real_correct / float(d_real.size(0) + d_fake.size(0))

        d_loss = d_loss_real + d_loss_fake
        loss = d_loss * weight
        return loss, {
            "loss": loss.clone().detach(),
            "d_loss": d_loss.clone().detach(),
            "dis_acc": real_acc.clone().detach(),
            "dis_correct": real_correct.clone().detach(),
            "loss_real": d_loss_real.clone().detach(),
            "loss_fake": d_loss_fake.clone().detach(),
        }
    
    elif gan.lower() == "ls":
        mse = nn.MSELoss()
        B = d_fake.size(0)

        real_label_np = np.ones((B,))
        fake_label_np = np.zeros((B,))

        # if noise_label:
        real_label_np = smooth_labels(B,ran=[0.9,1.0])
            #fake_label_np = smooth_labels(B,ran=[0.0,0.1])
            # occasionally flip the labels when training the D to
            # prevent D from becoming too strong
        real_label_np = noisy_labels(real_label_np, 0.05)
            #fake_label_np = noisy_labels(fake_label_np, 0.05)


        real_label = torch.from_numpy(real_label_np.astype(np.float32)).cuda()
        real_label = real_label.reshape(real_label.shape[0],1)
        fake_label = torch.from_numpy(fake_label_np.astype(np.float32)).cuda()
        fake_label = fake_label.reshape(fake_label.shape[0],1)
        

        # real_label = Variable((1.0 - 0.9) * torch.rand(d_fake.size(0)) + 0.9).cuda()
        # fake_label = Variable((0.1 - 0.0) * torch.rand(d_fake.size(0)) + 0.0).cuda()

        t = 0.5
        real_correct = (d_real >= t).float().sum()
        real_acc = real_correct / float(d_real.size(0))

        fake_correct  = (d_fake < t).float().sum()
        fake_acc = fake_correct / float(d_fake.size(0))
        # + d_fake.size(0))

        # real_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(1).cuda())
        # fake_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(0).cuda())

        g_loss = F.mse_loss(d_fake, fake_label)
        d_loss = F.mse_loss(d_real, real_label)

        if d_real_p is not None and d_fake_p is not None:

            real_label_p = Variable((1.0 - 0.9) * torch.rand(d_fake_p.size(0), d_fake_p.size(1)) + 0.9).cuda()
            fake_label_p = Variable((0.1 - 0.0) * torch.rand(d_fake_p.size(0), d_fake_p.size(1)) + 0.0).cuda()
         

            # real_label_p = Variable(torch.FloatTensor(d_real_p.size(0), d_real_p.size(1)).fill_(1).cuda())
            # fake_label_p = Variable(torch.FloatTensor(d_real_p.size(0), d_real_p.size(1)).fill_(0).cuda())
            g_loss_p = F.mse_loss(d_fake_p, fake_label_p)
            d_loss_p = F.mse_loss(d_real_p, real_label_p)

            g_loss = (g_loss + 0.1*g_loss_p)
            d_loss = (d_loss + 0.1*d_loss_p)

        loss =  (g_loss+d_loss)/2.0
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach(),
            'd_loss': g_loss.clone().detach(),
            "fake_acc": fake_acc.clone().detach(),
            "real_acc": real_acc.clone().detach()
        }
    elif gan.lower() =="gan":

        d_real_target = torch.tensor([1.0]).cuda()
        d_fake_target = torch.tensor([0.0]).cuda()
        discriminator_loss = functools.partial(BCEloss, d_real_target=d_real_target, d_fake_target=d_fake_target)

        g_loss, d_loss = discriminator_loss(d_fake, d_real)

        if d_real_p is not None and d_fake_p is not None:
            g_loss_p,d_loss_p = discriminator_loss(d_fake_p.view(-1),d_real_p.view(-1))
            g_loss = (g_loss + g_loss_p)/2.0
            d_loss = (d_loss + d_loss_p)/2.0

        loss =  (g_loss+d_loss)/2.0
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach(),
            'd_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "real":
        y = Variable(torch.Tensor(d_real.size(0)).fill_(1.0), requires_grad=False)
        d_loss = torch.mean((d_real - torch.mean(d_fake) - y) ** 2)
        g_loss = torch.mean((d_fake - torch.mean(d_real) + y) ** 2)
        loss =  (g_loss+d_loss)/2.0

    else:
        raise NotImplementedError("Not implement: %s" % gan)


def gen_loss(d_real, d_fake,gan="wgan", weight=1., d_real_p=None, d_fake_p=None,noise_label=False):
    if gan.lower() == "wgan":
        wg_loss_orig = - d_fake.mean()
        wg_loss = wg_loss_orig * weight
        return wg_loss, {
            "wgan_gen_loss": wg_loss.clone().detach().item(),
            "wgan_gen_loss_orig": wg_loss_orig.clone().detach().item(),
        }
    elif gan.lower() == "hinge":
        g_loss = -d_fake.mean()
        d_correct = (d_real >= 0.).float().sum() + (d_fake < 0.).float().sum()
        d_acc = d_correct / float(d_real.size(0) + d_fake.size(0))

        loss = weight * g_loss
        return loss, {
            'loss': loss.clone().detach(),
            "dis_acc": d_acc.clone().detach(),
            "dis_correct": d_correct.clone().detach(),
            'g_loss': g_loss.clone().detach()
        }
   
    elif gan.lower() == "ls":
        #mse = nn.MSELoss()
        B = d_fake.size(0)
        #real_label_np = np.ones((B,))
        fake_label_np = np.ones((B,))
      

        if noise_label:
            # occasionally flip the labels when training the generator to fool the D
            fake_label_np = noisy_labels(fake_label_np, 0.05)

        #real_label = torch.from_numpy(real_label_np.astype(np.float32)).cuda()

        fake_label = torch.from_numpy(fake_label_np.astype(np.float32)).cuda()
        fake_label = fake_label.reshape(fake_label.shape[0],1)
        
      
        # real_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(1).cuda())
        # fake_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(0).cuda())
        g_loss = F.mse_loss(d_fake, fake_label)


        if d_fake_p is not None:
            fake_label_p = Variable(torch.FloatTensor(d_fake_p.size(0), d_fake_p.size(1)).fill_(1).cuda())
            g_loss_p = F.mse_loss(d_fake_p,fake_label_p)
            g_loss = g_loss + 0.2*g_loss_p

        loss = weight * g_loss
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "gan":

        # fake_target = torch.tensor([1.0]).cuda()
        fake_target = torch.tensor([1.0]).cuda()
        fake_loss = functools.partial(BCEfakeloss, target=fake_target)
        g_loss = fake_loss(d_fake)
  
        if d_fake_p is not None:
            g_loss_p = fake_loss(d_fake_p.view(-1))
            g_loss = g_loss + g_loss_p

        # loss = weight * g_loss
        loss = weight * g_loss
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "real":
        # https://github.com/weishenho/SAGAN-with-relativistic/blob/master/main.py
        y = Variable(torch.Tensor(d_real.size(0)).fill_(1.0), requires_grad=False)
        d_loss =  torch.mean((d_real - torch.mean(d_fake) + y) ** 2)
        g_loss = torch.mean((d_fake - torch.mean(d_real) - y) ** 2)

        # d_loss = torch.mean((d_real - torch.mean(d_fake) - y) ** 2)
        # g_loss = torch.mean((d_fake - torch.mean(d_real) + y) ** 2)
        loss = (g_loss + d_loss) / 2.0

    else:
        raise NotImplementedError("Not implement: %s" % gan)


def get_edge_features(x, k, num=-1, idx=None, return_idx=False):
    """
    Args:
        x: point cloud [B, dims, N]
        k: kNN neighbours
    Return:
        [B, 2dims, N, k]    
    """
    B, dims, N = x.shape

    # batched pair-wise distance
    if idx is None:
        xt = x.permute(0, 2, 1)
        xi = -2 * torch.bmm(xt, x)
        xs = torch.sum(xt**2, dim=2, keepdim=True)
        xst = xs.permute(0, 2, 1)
        dist = xi + xs + xst # [B, N, N]

        # get k NN id
        _, idx_o = torch.sort(dist, dim=2)
        idx = idx_o[: ,: ,1:k+1] # [B, N, k]
        idx = idx.contiguous().view(B, N*k)


    # gather
    neighbors = []
    for b in range(B):
        tmp = torch.index_select(x[b], 1, idx[b]) # [d, N*k] <- [d, N], 0, [N*k]
        tmp = tmp.view(dims, N, k)
        neighbors.append(tmp)

    neighbors = torch.stack(neighbors) # [B, d, N, k]

    # centralize
    central = x.unsqueeze(3) # [B, d, N, 1]
    central = central.repeat(1, 1, 1, k) # [B, d, N, k]

    ee = torch.cat([central, neighbors-central], dim=1)
    assert ee.shape == (B, 2*dims, N, k)

    if return_idx:
        return ee, idx
    return ee


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc,axis=0)
    pc = pc -centroid
    m = np.max(np.sqrt(np.sum(pc**2,axis=1)))
    pc = pc/m
    return pc 

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def normalize_point_cloud(inputs):
    """
    input: pc [N, P, 3]
    output: pc, centroid, furthest_distance
    """
    #print("shape",input.shape)
    # C = inputs.shape[-1]
    # pc = inputs[:,:,:3]
    # if C > 3:
    #     nor = inputs[:,:,3:]

    pc = inputs
    centroid = np.mean(pc, axis=1, keepdims=True)
    pc = pc - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(pc ** 2, axis=-1, keepdims=True)), axis=1, keepdims=True)
    pc = pc / furthest_distance
    # if C > 3:
    #     return np.concatenate([pc,nor],axis=-1)
    # else:
    return pc


class scalePointCloud(object):
    def __init__(self,low =0.8, high=1.25):
        self.low,self.high = low,high
    
    def __call__(self,points):
        scaler = np.random.uniform(self.low,self.high)
        points[:] *= scaler
        return points 

class toTensorPointCloud(object):
    def __call__(self,points):
        return torch.from_numpy(points).float()

class gaussianNoisePointCloud(object):
    def __init__(self, mu=0,sigma=1):
        self.mu = mu
        self.sigma = sigma
    
    def __call__(self,points):
        gnoise = np.random.normal(self.mu,self.sigma,points.shape[0])
        gnoise = np.tile(gnoise,(3,1)).T
        points += gnoise
        return points
