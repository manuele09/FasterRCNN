from dataset import *
from averager import Averager
from model import FasterModel
from utils import collate_fn

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

BATCH_SIZE = 1  
NUM_WORKERS = 1

base_path = ".."
# #################### DATASET ####################
# #Define which dataset to use. If you use the code as it is
#the dataset will be automatically downloaded if not present.
image_base_path = base_path + "/dataset"
list_file_name = "valid.real.txt" #Real Validation Dataset
# list_file_name = "train.real.txt" #Real Training Dataset
# list_file_name = "valid.virtual.txt" #Virtual Validation Dataset
# list_file_name = "train.virtual.txt" #Virtual Training Dataset

transform = transforms.Compose([transforms.ToTensor()])
dataset = RealDataset(image_base_path, list_file_name=list_file_name, download_dataset=True, remove_unused_dataset=False, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)

#If you want to see the datasets image and bounding.
dataset.show_bounding(0)

# #################### PARAMETERS ####################
# load_pretrained_model = False #If True, load the model parameters
# load_from_wandb = False #If True, load the model from wandb, else load from local
# load_epoch = 0 #Epoch to load from
# load_batch = 0 #Batch to load from

# resume_training = False #If True, resume training from a checkpoint
# resume_batch = 0 #Batch to resume training from (checkpoint)

# log_to_wandb = False #If True, log the model parameters and loss to wandb

# if load_pretrained_model:
#   load_dict = {"load_from_wandb": load_from_wandb,
#                "wandb_entity": "emacannizzo",
#                "wandb_project": "Shuffled Virtual Dataset",
#               "epoch": load_epoch,
#               "batch": load_batch
#                }
# else:
#   load_dict = None

# if resume_training:
#   items_to_skip = resume_batch*BATCH_SIZE
# else:
#   items_to_skip = 0

# if log_to_wandb:
#   wandb_dict = {"wandb_api_key": "e3f943e00cb1fa8a14fd0ea76ed9ee6d50f86f5b",
#               "wandb_entity": "emacannizzo",
#               "wandb_project": "Shuffled Virtual Dataset"}
# else:
#   wandb_dict = None
# ####################################################

# dataset.skip_items(items_to_skip)
# model = FasterModel(dataloader, base_path, load_dict=load_dict)

#model.train(print_freq=1, save_freq=1)

# model.evaluate(dataloader)


# im, target = dataset[1]

# model.apply_object_detection(im)