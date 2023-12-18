import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from piq import FID, SSIMLoss
import torchvision
from IPython.display import HTML
from src.models.dcgan import Generator



def to_zero_one_tensor(ten):
    max_ten = ten.max()
    min_ten = ten.min()
    return (ten - min_ten) / (max_ten - min_ten)

def create_gen_folder(gen, nz, device, const_z):
    image_size=64

    dataset = dset.ImageFolder(root='test_data/',
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                         shuffle=False, num_workers=2, drop_last=False)

    real_ims = []
    gen_ims = []
    used = 0
    for data in dataloader:
        ims = data[0]
        real_ims.append(ims.detach())
        z = const_z[used:used+ims.shape[0], ...]
        fake_im = gen(z).detach()
        gen_ims.append(fake_im)
        used += ims.shape[0]

    real_ims = torch.cat(real_ims)
    gen_ims = torch.cat(gen_ims)
    real_ims = to_zero_one_tensor(real_ims)
    gen_ims = to_zero_one_tensor(gen_ims)


    for i in range(gen_ims.shape[0]):
        im = gen_ims[i]
        torchvision.utils.save_image(im, f'gen_data/data/{i+1}.png')
    ssim = SSIMLoss(data_range=1.)
    return ssim(real_ims.detach().cpu(), gen_ims.detach().cpu())


def collator(data):
    ims = []
    labels = []
    for d in data:
        ims.append(d[0])
        labels.append(d[1])
    b_ims = torch.stack(ims, dim=0)
    labels = torch.tensor(labels)
    return {'images': b_ims, 
            'labels': labels}

def FIDCalc(fid_metric):

    image_size = 64

    real_ds = dset.ImageFolder(root='test_data/',
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                           ]))
    fake_ds = dset.ImageFolder(root='gen_data/',
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                           ]))
    real_dataloader = torch.utils.data.DataLoader(real_ds, batch_size=32,
                                         shuffle=False, num_workers=2, drop_last=False, collate_fn=collator, pin_memory=True)
    fake_dataloader = torch.utils.data.DataLoader(fake_ds, batch_size=32,
                                         shuffle=False, num_workers=2, drop_last=False, collate_fn=collator, pin_memory=True)





    

    s1 = fid_metric.compute_feats(real_dataloader)
    s2 = fid_metric.compute_feats(fake_dataloader)

    fid = fid_metric(s1, s2)
    return fid