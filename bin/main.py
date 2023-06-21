from dataset import *
from custom_utils import *
from averager import Averager
from model import FasterModel

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import subprocess
import sys

# https://github.com/cocodataset/cocoapi/tree/master/
# cd coco/PythonAPI
# make
# sudo make install
# sudo python setup.py install

# before doing above steps install cython

BATCH_SIZE = 2 # increase / decrease according to GPU memeory
RESIZE_TO = 1000 # resize the image for training and transforms
NUM_EPOCHS = 1 # number of epochs to train for
NUM_WORKERS = 2

base_path = ".."

image_base_path = base_path + "/real_dataset"
list_path = image_base_path + "/train/list.txt"
dir_to_mantain = 1

# image_base_path = base_path + "/virtual_dataset"
# list_path = image_base_path + "/train.virtual.txt"
# dir_to_mantain = 2

transform = transforms.Compose([transforms.ToTensor()])

train_list = modify_list(list_path, dir_to_mantain, image_base_path)


#dataset = RealDataset(image_base_path, images_list=train_list, transform=transform)

#download case
dataset = RealDataset(image_base_path, list_file_name="train.virtual.txt", transform=transform, download_virtual_dataset=True)

#dataset.show_bounding(1)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)


model = FasterModel(base_path)
# model.load_model(0, 7)

model.train(dataloader, 1, save_freq=1)

# model.evaluate(dataloader)

