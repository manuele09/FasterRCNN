import torch
from torch.utils.data.dataset import Dataset

from PIL import Image
from os import path
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import shutil
import subprocess
import zipfile

from custom_utils import change_extension, from_path_to_names

#takes as input a list_file, containing the paths of the images.
#modify the paths in the list_file, in order to mantain n_dirs_to_mantain directories
#and adding new_root_path as root.

def modify_list(list_file, n_dirs_to_mantain, new_root_path):
    #Example of name in input_file: /dataset_1088x612/27_03_19_19_15_32/000/000001.png
    #n_dirst_to_mantain: 2
    #new_root_path: /Root
    #Output name: /Root/27_03_19_19_15_32/000/000001.png
    new_list = []
    with open(list_file, "r") as f:
        file_paths = f.readlines()
        for i in range(len(file_paths)):
            dirs, file_name = os.path.split(file_paths[i].replace('\\', os.sep))
            for j in range(n_dirs_to_mantain):
                dirs, d = os.path.split(dirs)
                file_name = os.path.join(d, file_name)
            file_name = os.path.join(new_root_path, file_name)
            new_list.append(file_name.strip())
    return new_list


class RealDataset(Dataset):
    def scan_base_path(self):
        dirs = [name for name in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, name))]
        return dirs

    #return a dictionary with the following keys: "image", "target".
    def read_target_from_file(self, index, im_width, im_height):
        #bounding file format: x_center, y_center, width, height (all expressed in percentages)
        #bounding new format: x_min, y_min, x_max, y_max (expressed in the real image range)
        labels = []
        bounding_boxes = []

        target_path = change_extension(self.images_list[index], ".txt")
        with open(target_path, 'r') as file:
            for line in file:
                target_str = line.strip().split() #label     x_center  y_center  width     height
                                              #target[0] target[1] target[2] target[3] target[4]
                target = [float(x) for x in target_str]
                
                bbox = []

                #x_min
                bbox.append((target[1] - target[3] / 2)*im_width) #x_c - w / 2
                #y_min
                bbox.append((target[2] - target[4] / 2)*im_height) #y_c - h / 2
                #x_max
                bbox.append((target[1] + target[3] / 2)*im_width) #x_c + w / 2
                #y_max
                bbox.append((target[2] + target[4] / 2)*im_height) #y_c + h / 2

                bounding_boxes.append(bbox)
                labels.append(int(target[0]))
            
        target_dict = {}
        target_dict["boxes"] = torch.tensor(bounding_boxes)
        target_dict["labels"] = torch.tensor(labels)
        target_dict["image_id"] = torch.tensor([index])
        target_dict["area"] = (target_dict["boxes"][:, 3] - target_dict["boxes"][:, 1]) * (target_dict["boxes"][:, 2] - target_dict["boxes"][:, 0])
        target_dict["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)


        return target_dict

    #images_root: path containg all the images
    #image_list_path: path of the file containing the list of all the images name
    def __init__(self, images_list, base_path, transform=None, download_dataset=False, dirs_ids=None):
        self.base_path = base_path
        self.download_dataset = download_dataset

        self.images_list = images_list
        self.transform = transform

        self.str_label = ["No Elmet", "Elmet", "Welding Mask", "Ear Protection", "No Gilet", "Gilet", "Person"]
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', "brown"]

        if self.download_dataset:
            if dirs_ids is None:
                print("Error: dirs_ids is None")
                exit()
            self.downloaded_dirs = self.scan_base_path()
            self.dirs_ids = dirs_ids

    def __getitem__(self, index):
        if self.download_dataset:
            #Estraggo il nome della sottorcartella che contiene l'immagine
            current_dir = self.images_list[index].split(os.path.sep)[-3] 

            #rimuovo tutte le cartelle in eccesso
            for i in range(len(self.downloaded_dirs)):
                if self.downloaded_dirs[i] != current_dir:
                    shutil.rmtree(os.path.join(self.base_path, self.downloaded_dirs[i]))

            #verifico se la cartella current_dir è presente o no
            if current_dir not in self.downloaded_dirs:
                #Scarico il file zip
                print(f"Downloading {current_dir}...")
                file_id = self.dirs_ids[current_dir]
                file_name = os.path.join(self.base_path, current_dir) + ".zip"
                download_url = f"https://studentiunict-my.sharepoint.com/:u:/g/personal/cnnmnl01r09m088w_studium_unict_it/{file_id}?download=1"
                command = f"wget -c --no-check-certificate -O {file_name} {download_url} > /dev/null 2>&1"
                print(command)
                subprocess.run(command, shell=True)
                print("Download completed.")

                #Estrapolo il file zip
                print(f"Unzipping {current_dir}.zip ...")
                with zipfile.ZipFile(file_name, 'r') as zip_ref:
                    zip_ref.extractall(self.base_path)
                print("Unzip completed.") 

                #Elimino la zip
                os.remove(file_name)           

            self.downloaded_dirs = [current_dir]

            exit()


        if (index >= len(self.images_list)): #DA RIVEDERE
            return self.__getitem__(index % len(self.images_list))
        try:
            im = Image.open(self.images_list[index]).convert('RGB')
        except Exception as e:
            print(" Exception caught: skipping dataset element.")
            print(f"Error reading image {self.images_list[index]}: {e}")
            del self.images_list[index]
            return self.__getitem__(index)
        
        if self.transform is not None:
            im = self.transform(im)
        
        try:
            target = self.read_target_from_file(index, im.shape[2], im.shape[1])
            if target["boxes"].shape[0] == 0 or target["boxes"].shape[1] != 4:
                raise Exception("Invalid box format.")

        except Exception as e:
            print("Exception caught: skipping dataset element.")
            print(f"Error reading target of {self.images_list[index]}: {e}")
            del self.images_list[index]
            return self.__getitem__(index)
        
        return im, target
    
    def __len__(self):
        return len(self.images_list)
    
    def show_bounding(self, index):
        image, targets = self[index]
        self.label, self.bounding = targets["labels"], targets["boxes"]
        
        fig, ax = plt.subplots(1)
        ax.imshow(image.permute(1, 2, 0))
        #ax.imshow(image)
        
        for l, bbox in zip(self.label, self.bounding):
            width = bbox[2] - bbox[0]  
            height = bbox[3] - bbox[1] 

            x = bbox[0] 
            y = bbox[1] 
                        
            # Create a rectangle patch
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=self.colors[l], facecolor="none")

            # Add the rectangle to the axes
            ax.add_patch(rect)
            ax.text(x, y - 10, self.str_label[l], color=self.colors[l])

        # Show the plot
        plt.show()
    

