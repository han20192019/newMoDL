%matplotlib notebook
import os, sys
import logging
import random
import h5py
import shutil
import time
import argparse
import numpy as np
import sigpy.plot as pl
import torch
import sigpy as sp
import torchvision
from torch import optim
from tensorboardX import SummaryWriter
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
# import custom libraries
from utils import transforms as T
from utils import subsample as ss
from utils import complex_utils as cplx
from utils.resnet2p1d import generate_model
from utils.flare_utils import roll
# import custom classes
from utils.datasets import SliceData
from subsample_fastmri import MaskFunc
from MoDL_single import UnrolledModel
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
%load_ext autoreload
%autoreload 2

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class DataTransform:
    """
    Data Transformer for training unrolled reconstruction models.
    """

    def __init__(self, mask_func, args, use_seed=False):
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.rng = np.random.RandomState()

    def __call__(self, kspace, target, slice):
        im_lowres = abs(sp.ifft(sp.resize(sp.resize(kspace,(640,24)),(640,372))))
        magnitude_vals = im_lowres.reshape(-1)
        k = int(round(0.05 * magnitude_vals.shape[0]))
        scale = magnitude_vals[magnitude_vals.argsort()[::-1][k]]
        kspace = kspace/scale
        target = target/scale
        # Convert everything from numpy arrays to tensors
        kspace_torch = cplx.to_tensor(kspace).float()   
        target_torch = cplx.to_tensor(target).float()   
        mask_slice = np.ones((640,372))
        mk1 = self.mask_func((1,1,372,2))[0,0,:,0]
        knee_masks = mask_slice*mk1
        mask_torch = torch.tensor(knee_masks[...,None]).float()   
        kspace_torch = kspace_torch*mask_torch

        return kspace_torch,target_torch,mask_torch

def create_datasets(args):
    # Generate k-t undersampling masks
    train_mask = MaskFunc([0.08],[4])
    train_data = SliceData(
        root=str(args.data_path),
        transform=DataTransform(train_mask, args),
        sample_rate=1
    )
    return train_data
def create_data_loaders(args):
    train_data = create_datasets(args)
#     print(train_data[0])

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    return train_loader
def build_optim(args, params):
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer



#Hyper parameters
params = Namespace()
params.data_path = "train/"
params.batch_size = 2
params.num_grad_steps = 4
params.num_cg_steps = 8
params.share_weights = True
params.modl_lamda = 0.05
params.lr = 0.0001
params.weight_decay = 0
params.lr_step_size = 500
params.lr_gamma = 0.5
params.epoch = 200

train_loader = create_data_loaders(params)

single_MoDL = UnrolledModel(params).to(device)
optimizer = build_optim(params, single_MoDL.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, params.lr_step_size, params.lr_gamma)
criterion = nn.L1Loss()

for epoch in range(params.epoch):
    single_MoDL.train()
    avg_loss = 0.

    for iter, data in enumerate(train_loader):
        input,target,mask = data
        input = input.to(device)
        target = target.to(device)
        mask = mask.to(device)
        im_out = single_MoDL(input.float(),mask=mask)
        loss = criterion(im_out,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        if iter % 20 == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{params.epoch:3d}] '
                f'Iter = [{iter:4d}/{len(train_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g}'
            )
    #Saving the model
    exp_dir = "checkpoints/"
    torch.save(
        {
            'epoch': epoch,
            'params': params,
            'model': single_MoDL.state_dict(),
            'optimizer': optimizer.state_dict(),
            'exp_dir': exp_dir
        },
        f=os.path.join(exp_dir, 'model_%d.pt'%(epoch))
    )