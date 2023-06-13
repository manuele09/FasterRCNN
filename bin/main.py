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



BATCH_SIZE = 1 # increase / decrease according to GPU memeory
RESIZE_TO = 1000 # resize the image for training and transforms
NUM_EPOCHS = 1 # number of epochs to train for
NUM_WORKERS = 4

base_path = ".."
subprocess.run(["rm", "-rf", base_path + "/logs"])

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(( RESIZE_TO, RESIZE_TO), antialias=True)
                               , transforms.Lambda(remove_alpha_channel)])

modify_names_list(base_path + "/real_dataset/train/list.txt", base_path + "/real_dataset/train/list_v2.txt")

dataset = RealDataset(base_path + "/real_dataset/train", base_path + "/real_dataset/train/list_v2.txt", transform=transform)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)

#dataset.show_bounding(54)
full_model = FasterModel(base_path, last_epoch=-1, last_batch=-1)





train_loss_hist = Averager()
for i in range(NUM_EPOCHS):
    full_model.train(dataloader, train_loss_hist)











