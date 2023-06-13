import torch
from torch.utils.data.dataset import Dataset

from PIL import Image
from os import path
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def transform_extension(filename, new):
    # Split the filename into name and extension
    name, extension = filename.rsplit(".", maxsplit=1)

    # Change the extension to ".txt"
    new_filename = name + new

    return new_filename

def modify_names_list(input_file, output_file):
    # Read the existing file and extract the file names
    with open(input_file, "r") as f:
        file_paths = f.readlines()
        file_names = [file_path.split("\\")[-1].strip() for file_path in file_paths]

    # Write the file names to the new file
    with open(output_file, "w") as f:
        for file_name in file_names:
            f.write(file_name + "\n")

#return a list of labels and a list of bounding boxes
def read_labels_from_file(file_path):
    labels = []
    bounding_boxes = []
    #file format: x_center, y_center, width, height
    #new format: x_min, y_min, x_max, y_max
    with open(file_path, 'r') as file:
        for line in file:
            label_info = line.strip().split()
            label = int(label_info[0])
            bounding_box = list(map(float, label_info[1:]))
            b = []

            b.append(bounding_box[0] - bounding_box[2] / 2) #x_min
            b.append(bounding_box[1] - bounding_box[3] / 2) #y_min

            b.append(bounding_box[0] + bounding_box[2] / 2) #x_max
            b.append(bounding_box[1] + bounding_box[3] / 2) #y_max
        
            labels.append(label)
            bounding_boxes.append(b)
    return labels, bounding_boxes

class RealDataset(Dataset):
    def __init__(self, image_path, image_list_path, transform=None):
        
        self.image_path = image_path #path containg all the images
        
        self.image_list_names = np.loadtxt(image_list_path,dtype=str) #list of all the images name

        self.transform = transform

        self.str_label = ["No Elmet", "Elmet", "Welding Mask", "Ear Protection", "No Gilet", "Gilet", "Person"]
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', "brown"]
    
    def __getitem__(self, index):
        f = self.image_list_names[index] #take the name of the image
        im = Image.open(path.join(self.image_path, f))
        
        if self.transform is not None:
            im = self.transform(im)
        
        return im, self.get_targets(index)
    
    def __len__(self):
        return len(self.image_list_names)

    #return a dictionary with the bounding boxes and the labels of the image[index]
    def get_targets(self, index):
        self.label, self.bounding = read_labels_from_file(path.join(self.image_path, transform_extension(self.image_list_names[index], ".txt")))
        l_list = []
        b_list = []
        for l, box in zip(self.label, self.bounding):
            l_list.append(l)
            b_list.append(box)
        d = {}
        d["boxes"] = torch.tensor(b_list)
        d["labels"] = torch.tensor(l_list)
        return d
    
    
    def show_bounding(self, index):
        image, targets = self[index]
        self.label, self.bounding = targets["labels"], targets["boxes"]
        
        fig, ax = plt.subplots(1)
        ax.imshow(image.permute(1, 2, 0))
        #ax.imshow(image)
        
        for l, bbox in zip(self.label, self.bounding):
            width = (bbox[2] - bbox[0]) * image.shape[2]  
            height = (bbox[3] - bbox[1]) * image.shape[1] 

            x = bbox[0] * image.shape[2] 
            y = bbox[1] * image.shape[1] 
            
            # Create a rectangle patch
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=self.colors[l], facecolor="none")

            # Add the rectangle to the axes
            ax.add_patch(rect)
            ax.text(x, y - 10, self.str_label[l], color=self.colors[l])

        # Show the plot
        plt.show()
    

