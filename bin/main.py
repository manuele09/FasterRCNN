from dataset import *
from model import FasterModel
from utils import collate_fn

from torch.utils.data import DataLoader

from torchvision import transforms

import matplotlib.pyplot as plt


BATCH_SIZE = 1  
NUM_WORKERS = 1 #must be set to one if the daset is set to be downloaded


base_path = ".."
# #################### DATASET ####################
# #Define which dataset to use. If you use the following code as it is
#the dataset will be automatically downloaded if not present.
image_base_path = base_path + "/dataset" 
#Chose one of the following names that define wich dataset to use
# list_file_name = "valid.real.txt" #Real Validation Dataset
# list_file_name = "train.real.txt" #Real Training Dataset
list_file_name = "valid.virtual.txt" #Virtual Validation Dataset
# list_file_name = "train.virtual.txt" #Virtual Training Dataset

transform = transforms.Compose([transforms.ToTensor()])
dataset = RealDataset(image_base_path, list_file_name=list_file_name, download_dataset=True, remove_unused_dataset=False, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)

#If you want to see the datasets image and bounding.
dataset.show_bounding(0)

# #################### PARAMETERS ####################
load_pretrained_model = True #If True, load the model parameters
load_from_wandb = True #If True, load the model from wandb, else load from local
load_epoch = 39 #Epoch to load from
load_batch = 0 #Batch to load from

#The three projects created in wandb are the following:
# load_model_name = "Virtual-Dataset" #Model trained with Virtual Dataset
load_model_name = "Real Dataset" #Model trained with Real Dataset
# load_model_name = "Virtual Model on Real Dataset" #Model trained with Virtual Dataset and Real Dataset

#WARNING: Not all combinations of load_epoch/batch/model_name can be utilized. A list of the
#avaible combinations can be found at the end of this file.

resume_training = False #If True, resume training from a checkpoint
resume_batch = 0 #Batch to resume training from (checkpoint)

log_to_wandb = False #If True, log the model parameters and loss to wandb
project_log_wandb = "Virtual-Dataset" #Project to log to
wandb_api_key = "" #API key to log to wandb

#Setting a dictionary with the parameters to load the model 
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
  wandb_dict = {"wandb_api_key": wandb_api_key,
              "wandb_entity": "emacannizzo",
              "wandb_project": project_log_wandb}
else:
  wandb_dict = None
# ####################################################

dataset.skip_items(items_to_skip) #Skip items if resuming training
model = FasterModel(dataloader, base_path, wandb_logging=wandb_dict, load_dict=load_dict)
#In the case of model downloaded from wandb, if the download was interrupted, it is necessary
#to manually delete the file before trying to run the code again.
#(It is displayed no progress bar for the download)
#Moreover, if is it asked to select an option from three options, you can select the third one.


# #################### TRAINING ####################
#scaler = torch.cuda.amp.GradScaler(enabled=False)
# model.train(print_freq = 2, save_freq=None, scaler=scaler)

# #################### EVALUATION ####################
# model.evaluate(dataloader, catIds = "all")

#Apply the model to an image to verify its predtions
index = 0
im, target = dataset[index]
fig = dataset.show_bounding(index)
model.apply_object_detection(im)

#necessary, else the plots will close immedietally.
plt.show()


################
#Here are the models obtained.

#Model trained with Virtual Dataset:
#    best for Virtual Validation: load_epoch=6, load_batch=0, load_model_name="Virtual-Dataset"
#    best for Real Validation: load_epoch=2, load_batch=0, load_model_name="Virtual-Dataset"

#Model trained with Real Dataset: load_epoch=39, load_batch=0, load_model_name="Real Dataset"

#Model trained with Virtual and Real Dataset: load_epoch=20, load_batch=0, load_model_name="Virtual Model on Real Dataset"