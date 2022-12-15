
import os
import pprint
pp = pprint.PrettyPrinter()
from datetime import datetime

from model.model_test import Model
from plyfile import PlyData
import numpy as np
from utilities import normalize_point_cloud
import torch
from scipy.optimize import linear_sum_assignment

# original author  https://github.com/ThibaultGROUEIX/AtlasNet 

def load_data(filename,num_points =2048):

    f = PlyData.read(filename)
    all_info = f['vertex'][:num_points]

    data = np.array([tuple(i) for i in all_info],dtype = object)
    result = np.array([tuple(i[:3]) for i in data])
    return normalize_point_cloud(result)

def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    x = x.double()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]

def emd_approx(x, y):
    bs, npts, mpts, dim = x.size(0), x.size(1), y.size(1), x.size(2)
    assert npts == mpts, "EMD only works if two point clouds are equal size"
    dim = x.shape[-1]
    x = x.reshape(bs, npts, 1, dim)
    y = y.reshape(bs, 1, mpts, dim)
    dist = (x - y).norm(dim=-1, keepdim=False)  # (bs, npts, mpts)

    emd_lst = []
    dist_np = dist.cpu().detach().numpy()
    for i in range(bs):
        d_i = dist_np[i]
        r_idx, c_idx = linear_sum_assignment(d_i)
        emd_i = d_i[r_idx, c_idx].mean()
        emd_lst.append(emd_i)
    emd = np.stack(emd_lst).reshape(-1)
    emd_torch = torch.from_numpy(emd).to(x)
    return emd_torch


def EMD_CD(sample_pcs, ref_pcs, batch_size, accelerated_cd=False, reduced=True,accelerated_emd=False):
    
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []
    iterator = range(0, N_sample, batch_size)

    for b_start in iterator:
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]
   
        dl, dr = distChamfer(sample_batch, ref_batch)
        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))

        emd_batch = emd_approx(sample_batch, ref_batch)
        emd_lst.append(emd_batch)

    if reduced:
        cd = torch.cat(cd_lst).mean()
        emd = torch.cat(emd_lst).mean()
    else:
        cd = torch.cat(cd_lst)
        emd = torch.cat(emd_lst)

    results = {
        'MMD-CD': cd,
        'MMD-EMD': emd,
    }
    return results