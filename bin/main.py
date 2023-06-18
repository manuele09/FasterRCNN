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

# https://github.com/cocodataset/cocoapi/tree/master/
# cd coco/PythonAPI
# make
# sudo make install
# sudo python setup.py install

# before doing above steps install cython

BATCH_SIZE = 1 # increase / decrease according to GPU memeory
RESIZE_TO = 1000 # resize the image for training and transforms
NUM_EPOCHS = 1 # number of epochs to train for
NUM_WORKERS = 4

base_path = ".."
subprocess.run(["rm", "-rf", base_path + "/logs"])

transform = transforms.Compose([transforms.ToTensor()])

# transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(( RESIZE_TO, RESIZE_TO), antialias=True)
#                                , transforms.Lambda(remove_alpha_channel)])


dataset = RealDataset(base_path + "/real_dataset/train", transform=transform)
#dataset.show_bounding(54)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)

model = FasterModel(base_path)
model.load_model(0, 1)


lr_scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=3, gamma=0.1)

for i in range(NUM_EPOCHS):
    model.train(dataloader, 1, save_freq=1)
    lr_scheduler.step()











