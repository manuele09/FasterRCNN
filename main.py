from dataset import RealDataset
from custom_utils import *
from averager import Averager
from model import FasterModel

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import intel_extension_for_pytorch as ipex


import torchvision
from torchvision import transforms

from PIL import Image
from os import path
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import datetime



BATCH_SIZE = 1 # increase / decrease according to GPU memeory
RESIZE_TO = 1000 # resize the image for training and transforms
NUM_EPOCHS = 1 # number of epochs to train for
NUM_WORKERS = 4

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(( RESIZE_TO, RESIZE_TO), antialias=True)
                               , transforms.Lambda(remove_alpha_channel)])

# transform = transforms.Compose([transforms.ToTensor()
#                                , transforms.Lambda(remove_alpha_channel)])

dataset = RealDataset("real_dataset/train", "real_dataset/train/list_v2.txt", transform=transform)



dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)

#dataset.show_bounding(54)
full_model = FasterModel("models_saved", last_epoch=0, last_batch=2)





train_loss_hist = Averager()



for i in range(NUM_EPOCHS):
    full_model.train(dataloader, train_loss_hist)











