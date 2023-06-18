import torch
from torch.utils.data.dataset import Dataset

from PIL import Image
from os import path
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from custom_utils import change_extension, from_path_to_names



class RealDataset(Dataset):

    #return a dictionary with the following keys: "image", "target".
    def read_target_from_file(self, index, im_width, im_height):
        #bounding file format: x_center, y_center, width, height (all expressed in percentages)
        #bounding new format: x_min, y_min, x_max, y_max (expressed in the real image range)
        labels = []
        bounding_boxes = []

        target_path = path.join(self.images_root, change_extension(self.images_names[index], ".txt"))
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

        return target_dict

    #images_root: path containg all the images
    #image_list_path: path of the file containing the list of all the images name
    def __init__(self, images_root, transform=None):
        
        self.images_root = images_root 
        self.images_names = from_path_to_names(path.join(images_root, "list.txt"))

        self.transform = transform

        self.str_label = ["No Elmet", "Elmet", "Welding Mask", "Ear Protection", "No Gilet", "Gilet", "Person"]
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', "brown"]
    
    def __getitem__(self, index):
        try:
            im = Image.open(path.join(self.images_root, self.images_names[index])).convert('RGB')
        except Exception as e:
            print(" Exception caught: skipping dataset element.")
            print(f"Error reading image {self.images_names[index]}: {e}")
            del self.images_names[index]
            return self.__getitem__(index)
        
        if self.transform is not None:
            im = self.transform(im)
        
        try:
            target = self.read_target_from_file(index, im.shape[2], im.shape[1])
            if target["boxes"].shape[0] == 0 or target["boxes"].shape[1] != 4:
                raise Exception("Invalid box format.")

        except Exception as e:
            print("Exception caught: skipping dataset element.")
            print(f"Error reading target of {self.images_names[index]}: {e}")
            del self.images_names[index]
            return self.__getitem__(index)
        
        return im, target
    
    def __len__(self):
        return len(self.images_names)
    
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
    

