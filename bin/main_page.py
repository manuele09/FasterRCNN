import streamlit as st

from dataset import *
# from custom_utils import *
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
import io
import os

# https://github.com/cocodataset/cocoapi/tree/master/
# cd coco/PythonAPI
# make
# sudo make install
# sudo python setup.py install

# before doing above steps install cython

BATCH_SIZE = 1  # increase / decrease according to GPU memeory
NUM_EPOCHS = 1  # number of epochs to train for
NUM_WORKERS = 1

base_path = ".."


###################Scelta del modello da utilizzare
model_list = ["Placeholder", "Real Dataset", "Virtual Dataset", "Virtual Dataset, fine tuned on Real Dataset"]
best_epoch_list = ["Placeholder", "Real Dataset", "Virtual Dataset"]

st.write("Choose the model to use")
selected_model = st.selectbox('Trained with: ', model_list)
selected_epoch = best_epoch_list[0]
if selected_model == model_list[0]:
    exit()
elif (selected_model == model_list[1]) :
    load_project = "Real Dataset"
    load_epoch = 15
    load_batch = 0
elif (selected_model == model_list[2]):
    load_project = "Shuffled Virtual Dataset"
    selected_epoch = st.selectbox('Best mAP for: ', best_epoch_list)
    if (selected_epoch == best_epoch_list[0]):
        load_epoch = 2
        load_batch = 0
    else:
        load_epoch = 6
        load_batch = 0
else:
    load_project = "Virtual Model on Real Dataset"
    load_epoch = 15
    load_batch = 0

if (selected_epoch == best_epoch_list[0]):
    exit()

load_dict = {"load_from_wandb": True,
             "wandb_entity": "emacannizzo",
             "wandb_project": load_project,
            "epoch": load_epoch,
            "batch": load_batch,
             }
#il modello verrà definito una volta creato il dataloader
###################################################

#########################Scelta del dataset da utilizzare

st.write("Choose the dataset where to apply the model")
avaibles_datasets = ["Placeholder", "Real Dataset Train", "Real Dataset Valid"]
selected_dataset = st.selectbox('Dataset:', avaibles_datasets)

base_path = "."
image_base_path = base_path + "/dataset"
if (selected_dataset == avaibles_datasets[0]):
    exit()
elif (selected_dataset == avaibles_datasets[1]):
    list_file_name = "train.real.txt"
else:
    list_file_name = "valid.real.txt"


transform = transforms.Compose([transforms.ToTensor()])
dataset = RealDataset(image_base_path, list_file_name=list_file_name, transform=transform, download_dataset=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)

# ################################################################
# Carichiamo il modello
st.write("Inizio")
model = FasterModel(dataloader, base_path, load_dict=load_dict)

st.write("Finot")
###################################################
# #Scelta dell'immagine da utilizzare
# image_path = st.selectbox('Image File Name', dataset.images_list)
# image_index = dataset.images_list.index(image_path)
# ################################################################


# ################################################################




# with st.sidebar:
#     options = st.multiselect(
#         'Select the classes to show',
#         dataset.str_label,
#         dataset.str_label)
#     treshold = st.slider('Select Treshold to use to filter the predictions',
#                          min_value=0.0, max_value=1.0, value=0.5, step=0.1)


# col1, col2 = st.columns(2)
# with col1:
#     st.write("Ground Truth")
#     fig = dataset.show_bounding(image_index, options)
#     st.pyplot(fig)






# # model.train(print_freq=1, save_freq=1)

# # model.evaluate(dataloader)
# with col2:
#     st.write("Prediction")
#     im, target = dataset[image_index]
#     fig = model.apply_object_detection(
#         im, treshold=treshold, classes_to_show=options)
#     st.pyplot(fig)
