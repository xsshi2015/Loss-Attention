from __future__ import print_function, division
import numpy as np
import torch
from torch.autograd import Variable
import argparse
import os
import torch.nn.functional as F 

parser = argparse.ArgumentParser(description='PyTorch Deep Noisy Label')
parser.add_argument('-ch', '--checkpoint', metavar='DIR', help='path to checkpoint (default: ./checkpoint)', default='./checkpoint')
parser.add_argument('-w', '--workers', default=0, type=int, metavar='N', help='number of workers for data processing (default: 4)')
parser.add_argument('-nb', '--num_bits', default=10, type=int, metavar='N', help='Number of binary bits to train (default: 8)')

def split(imageData, sn=1000, v_sn=1000):
    trainData = np.array(imageData.data)
    trainLabel = np.squeeze(imageData.targets)

    u_label = np.squeeze(np.unique(trainLabel))
    train_idx = []
    dict_idx = []
    val_idx = []
    sn = int(sn/u_label.size)
    for iter in range(u_label.size):
        idx = np.squeeze(np.where(trainLabel==u_label[iter]))
        s_idx = np.squeeze(np.random.choice(idx, sn, replace=False))
        dict_idx.extend(s_idx)

        r_idx = np.squeeze(np.setdiff1d(idx, s_idx))
        if v_sn>0:
            srn = int(v_sn/u_label.size)
            sr_idx = np.squeeze(np.random.choice(r_idx, srn, replace=False))
            val_idx.extend(sr_idx)
            train_idx.extend(np.setdiff1d(r_idx, sr_idx))
        else:
            train_idx.extend(r_idx)

    train_idx = np.squeeze(train_idx)
    dict_idx = np.squeeze(dict_idx)
    val_idx = np.squeeze(val_idx)
    # import pdb; pdb.set_trace()

    return train_idx, dict_idx, val_idx


