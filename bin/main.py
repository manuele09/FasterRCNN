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
# dataset.show_bounding(0)

# #################### PARAMETERS ####################
load_pretrained_model = True #If True, load the model parameters
load_from_wandb = True #If True, load the model from wandb, else load from local
load_epoch = 20 #Epoch to load from
load_batch = 0 #Batch to load from
# load_model_name = "Virtual Dataset" #Model trained with Virtual Dataset
# load_model_name = "Real Dataset" #Model trained with Real Dataset
load_model_name = "Virtual Model on Real Dataset" #Model trained with Virtual Dataset and Real Dataset

resume_training = False #If True, resume training from a checkpoint
resume_batch = 0 #Batch to resume training from (checkpoint)

log_to_wandb = False #If True, log the model parameters and loss to wandb
project_log_wandb = "Virtual Dataset"

if load_pretrained_model:
  load_dict = {"load_from_wandb": load_from_wandb,
               "wandb_entity": "emacannizzo",
               "wandb_project": load_model_name,
              "epoch": load_epoch,
              "batch": load_batch
               }
else:
  load_dict = None

if resume_training:
  items_to_skip = resume_batch*BATCH_SIZE
else:
  items_to_skip = 0

if log_to_wandb:
  wandb_dict = {"wandb_api_key": "e3f943e00cb1fa8a14fd0ea76ed9ee6d50f86f5b",
              "wandb_entity": "emacannizzo",
              "wandb_project": "Shuffled Virtual Dataset"}
else:
  wandb_dict = None
# ####################################################

dataset.skip_items(items_to_skip)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)
model = FasterModel(dataloader, base_path, wandb_logging=wandb_dict, load_dict=load_dict)
#In the case of model downloaded from wandb, if the download was interrupted, it is necessary
#to manually delete the file before trying to run the code again.

scaler = torch.cuda.amp.GradScaler(enabled=False)
# model.train(print_freq = 2, save_freq=None, scaler=scaler)

# model.evaluate(dataloader, catIds = "all")

index = 0
im, target = dataset[index]
fig = dataset.show_bounding(index)
model.apply_object_detection(im)

plt.show()
