import streamlit as st
from constants import *
from dataset import *
from model import FasterModel
from utils import collate_fn

from torch.utils.data import DataLoader
from torchvision import transforms


#To use this demo install streamlit (pip), and run "streamlit run streamlit.py".
#It will create a local server with an interactive demo.    


BATCH_SIZE = 1  
NUM_EPOCHS = 1 
NUM_WORKERS = 1

base_path = ".."

###################Scelta del modello da utilizzare
model_list = ["Real Dataset", "Virtual Dataset", "Virtual Dataset, fine tuned on Real Dataset"]
best_epoch_list = ["Real Dataset", "Virtual Dataset"]

st.write("Choose the model to use")
selected_model = st.selectbox('Trained with: ', model_list)
if (selected_model == model_list[0]) :
    load_project = "Real Dataset"
    load_epoch = 39
    load_batch = 0
elif (selected_model == model_list[1]):
    load_project = "Virtual Dataset"
    selected_epoch = st.selectbox('Best mAP for: ', best_epoch_list)
    if (selected_epoch == best_epoch_list[0]):
        load_epoch = 2
        load_batch = 0
    else:
        load_epoch = 6
        load_batch = 0
else:
    load_project = "Virtual Model on Real Dataset"
    load_epoch = 20
    load_batch = 0

#Assumendo che è stato già scaricato
load_dict = {"load_from_wandb": False,
             "wandb_entity": "emacannizzo",
             "wandb_project": load_project,
            "epoch": load_epoch,
            "batch": load_batch,
             }

#########################Scelta del dataset da utilizzare

st.write("Choose the dataset where to apply the model")
avaibles_datasets = ["Real Dataset Train", "Real Dataset Valid"]
selected_dataset = st.selectbox('Dataset:', avaibles_datasets)

image_base_path = base_path + "/dataset"
if (selected_dataset == avaibles_datasets[0]):
    list_file_name = "train.real.txt"
else:
    list_file_name = "valid.real.txt"

transform = transforms.Compose([transforms.ToTensor()])
dataset = RealDataset(image_base_path, list_file_name=list_file_name, download_dataset=True, remove_unused_dataset=False, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)

st.write("Loading the dataset")
dataset.show_bounding(0)
st.write("Dataset loaded")

model = FasterModel(dataloader, base_path, wandb_logging=None, load_dict=load_dict)

# #Scelta dell'immagine da utilizzare
image_path = st.selectbox('Image File Name', dataset.images_list)
image_index = dataset.images_list.index(image_path)



with st.sidebar:
    options = st.multiselect(
        'Select the classes to show',
        str_label,
        str_label)
    treshold = st.slider('Select Treshold to use to filter the predictions',
                         min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    cat_ids = []
    for o in options:
        cat_ids.append(str_label.index(o))

col1, col2 = st.columns(2)
with col1:
    st.write("Ground Truth")
    fig = dataset.show_bounding(image_index, cat_ids, return_fig=True)
    st.pyplot(fig)


with col2:
    st.write("Prediction")
    im, target = dataset[image_index]
    fig = model.apply_object_detection(
        im, treshold=treshold, classes_to_show=cat_ids, return_fig=True)
    st.pyplot(fig)

