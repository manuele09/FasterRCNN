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


real_dataset = True
virtual_dataset = False
download_virtual = False
transform = transforms.Compose([transforms.ToTensor()])
if (real_dataset):
    image_base_path = base_path + "/real_dataset"
    list_path = image_base_path + "/train/list.txt"
    dir_to_mantain = 1
    train_list = modify_list(list_path, dir_to_mantain, image_base_path)
    dataset = RealDataset(
        image_base_path, images_list=train_list, transform=transform)
elif (virtual_dataset):
    image_base_path = base_path + "/virtual_dataset"
    list_path = image_base_path + "/train.virtual.txt"
    dir_to_mantain = 2
    train_list = modify_list(list_path, dir_to_mantain, image_base_path)
    dataset = RealDataset(
        image_base_path, images_list=train_list, transform=transform)
else:
    image_base_path = base_path + "/virtual_dataset"
    dataset = RealDataset(image_base_path, list_file_name="train.virtual.txt",
                          transform=transform, download_virtual_dataset=True)

# file_up = st.file_uploader("Upload an image", type="jpg")

image_path = st.selectbox('Image File Name', dataset.images_list)
image_index = dataset.images_list.index(image_path)


with st.sidebar:
    options = st.multiselect(
            'Select the classes to show',
            dataset.str_label,
            dataset.str_label)
    treshold = st.slider('Select Treshold to use to filter the predictions',
                          min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    
    
col1, col2 = st.columns(2)
with col1:
    st.write("Ground Truth")
    fig = dataset.show_bounding(image_index, options )
    st.pyplot(fig)









if True:
    resume_epoch = 4
    resume_batch = 0  # Non è il batch n, ma il batch n+1
    
    dataset.skip_items(resume_batch*BATCH_SIZE)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS, collate_fn=collate_fn)
    load_dict = {"load_from_wandb": False,
                "epoch": resume_epoch,
                "batch": resume_batch,
                "path": None}
else:
    load_dict = None
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS, collate_fn=collate_fn)
    
# @st.cache_data
def load_model():
    model = FasterModel(dataloader, base_path, load_dict=load_dict)
    return model

model = load_model()

#model.train(print_freq=1, save_freq=1)

# model.evaluate(dataloader)
with col2:
    st.write("Prediction")
    im, target = dataset[image_index]
    fig = model.apply_object_detection(im, treshold=treshold, classes_to_show=options)
    st.pyplot(fig)