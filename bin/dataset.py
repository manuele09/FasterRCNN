from constants import *

import torch
from torch.utils.data.dataset import Dataset

import os
import shutil
import subprocess
import zipfile

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches



#Change the extension of a filename (can include also a path)
# to new_ext
def change_extension(filename, new_ext):
    # Split the filename into name and extension
    name, extension = filename.rsplit(".", maxsplit=1)

    # Change the extension
    new_filename = name + new_ext
    return new_filename

#Read and retur a list from the path specified
def read_list_from_file(path):
    l = []
    try:
        with open(path, 'r') as file:
            for row in file:
                item = row.strip()  
                l.append(item)
    except FileNotFoundError:
        print("File not found. Check the path.")
    return l

#Returns the path of the file with the specified name
#inside the path. None if it is not in the path.
def find_path(name, path):
    for root, dirs, files in os.walk(path):
        if name in dirs or name in files:
            return os.path.join(root, name)
    return None

#Return a list of directories inside the path
def scan_path(path):
    dirs = [os.path.join(path, name) for name in os.listdir(
        path) if os.path.isdir(os.path.join(path, name))]
    return dirs


class RealDataset(Dataset):

    # return a dictionary with the following keys: "image", "target".
    # Needs the index, the width and the height of the image. 
    def read_target_from_file(self, index, im_width, im_height):
        # bounding file format: x_center, y_center, width, height (all expressed in percentages)
        # bounding new format: x_min, y_min, x_max, y_max (expressed in the real image range)
        labels = []
        bounding_boxes = []

        target_path = change_extension(self.images_list[index], ".txt")
        with open(target_path, 'r') as file:
            for line in file:
                target_str = line.strip().split()
                target = [float(x) for x in target_str]
                #Here is the content of target:
                # target[0] target[1] target[2] target[3] target[4]
                # label     x_center  y_center  width     height

                bbox = []

                # x_min
                bbox.append((target[1] - target[3] / 2)
                            * im_width)  # x_c - w / 2
                # y_min
                bbox.append((target[2] - target[4] / 2)
                            * im_height)  # y_c - h / 2
                # x_max
                bbox.append((target[1] + target[3] / 2)
                            * im_width)  # x_c + w / 2
                # y_max
                bbox.append((target[2] + target[4] / 2)
                            * im_height)  # y_c + h / 2

                bounding_boxes.append(bbox)
                labels.append(int(target[0]))

        target_dict = {}
        target_dict["boxes"] = torch.tensor(bounding_boxes)
        target_dict["labels"] = torch.tensor(labels)
        target_dict["image_id"] = torch.tensor([index])
        target_dict["area"] = (target_dict["boxes"][:, 3] - target_dict["boxes"]
                               [:, 1]) * (target_dict["boxes"][:, 2] - target_dict["boxes"][:, 0])
        target_dict["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)

        return target_dict


    #base_path: path of the directory that will contain the dataset
    #list_file_name: name of the file that contains the list of images paths.
    #   It must be inside the base_path directory.
    #download_dataset: if True, the dataset will be downloaded. To choose the dataset to download,
    #   set the list_file_name to one of this names: "train.virtual.txt", "valid.virtual.txt", "train.real.txt", "valid.real.txt".
    #   They will be downloaded automatically.
    #remove_unused_dataset: if True, all the directories that are not currently used by the 
    #   dataset will be removed, to save memory.
    #download_timeout: timeout to use when downloading from cloud
    #transform: the transformation to apply to the images.
    def __init__(self, base_path, list_file_name, download_dataset=False, remove_unused_dataset=True, download_timeout=None, transform=None):
        self.base_path = base_path
        self.download_dataset = download_dataset
        self.remove_unused_dataset = remove_unused_dataset
        self.download_timeout = download_timeout
        self.transform = transform
        self.images_list = []
        
        self.str_label = str_label
        self.colors = colors_bounding

        if self.download_dataset:
            #datasets will be a list of dictionaries, each one containing the 
            #name of the zip file saved in the cloud and its unique id (needed to download it)
            self.datasets = []
            self.datasets.append(virtual_dataset_train)
            self.datasets.append(virtual_dataset_valid)
            self.datasets.append(real_dataset_train)
            self.datasets.append(real_dataset_valid)

            #Given the list_file_name, find wich of the 4 datasets contains it,
            #and save it in choosen_dataset
            self.choosen_dataset = None
            for dataset in self.datasets:
                if list_file_name in dataset:
                    self.choosen_dataset = dataset
                    break
            if self.choosen_dataset is None:
                print(f"Error: The dataset corresponding to {list_file_name} was not found.")
                return
        
            # create the base_path if not exists
            if not os.path.exists(self.base_path):
                print("Creating base_path")
                os.makedirs(self.base_path)
                #download the txt file that contains the list of images
                self.download_and_extract(list_file_name, is_dir=False)

            # if the base_path exists but the list_file doesn't, download it
            if find_path(list_file_name, self.base_path) is None:
                self.download_and_extract(list_file_name, is_dir=False)

        #At this point there will be a file called f"{list_file_name}" in the base_directory,
        #regardless if it was downloaded or not.

        #Read the file, and modifies the relative paths to include the base_path.
        self.images_list = read_list_from_file(os.path.join(self.base_path, list_file_name))
        for i in range(0, len(self.images_list)):
            self.images_list[i] = os.path.join(self.base_path, self.images_list[i])

        if len(self.images_list) == 0:
            print("Error: images_list is void")
            return
        
        #An original copy of the images_list is saved, because, if we change momentarily 
        # the images_list we can restore it later (see skip_items())
        self.original_images_list = self.images_list.copy()

    # Skip the first start_index items of the dataset. Useful to resume training from a checkpoint.
    # Each time that is called, skips the items from the original list.
    def skip_items(self, start_index):
        self.images_list = self.original_images_list[start_index:]
        print(f"Skipping {start_index} items")

    # Downloads the file_name.zip from the shared link, extracts it in the base_path
    # File name is only the key in self.choosen_dataset, it is not necesserary how the
    #   file is called in the cloud
    #is_dir: boolean. If the zip to download contains a directory or a file.
    #show_progress: boolean. If True show the default progress bar created by wget.
    #Can be useful to make it False in a notebook, because each update of the progress bar
    #will be printed in a new line.
    def download_and_extract(self, file_name, is_dir, show_progress=True):
        # Download the zip
        print(f"Downloading {file_name}.zip ...")
        file_id = self.choosen_dataset[file_name]
        out_path = os.path.join(self.base_path, file_name) + ".zip"
        download_url = f"https://studentiunict-my.sharepoint.com/:u:/g/personal/cnnmnl01r09m088w_studium_unict_it/{file_id}?download=1"
        if show_progress:
            command = f"wget --no-check-certificate -O {out_path} {download_url}"
        else:
            command = f"wget --no-check-certificate -O {out_path} {download_url} > /dev/null 2>&1"
        
        while(True):
            try:
                #Here is the true download
                subprocess.run(command, shell=True, timeout=self.download_timeout)
                break
            except subprocess.TimeoutExpired:
                print("Timeout expired, retrying...")
                timeout += 10
                
        print("Download completed.")

        # Extract the zip
        print(f"Unzipping {file_name}.zip ...")
        if is_dir:
          save_path = os.path.join(self.base_path, file_name)
        else:
          save_path = self.base_path
        with zipfile.ZipFile(out_path, 'r') as zip_ref:
            zip_ref.extractall(save_path)
        print("Unzip completed.")

        # Delete the zip
        os.remove(out_path)

    #Returns the image and the target of the index-th element of the dataset.
    #If the image or the target are corrupted, they are removed and the next element is returned.
    #If the daset was downloaded, it will seemlessly download the next directory when needed.
    #No further action is needed by the user.
    def __getitem__(self, index):
        # To avoid index out of range (it is possible that some images are corrupted and removed)
        if (index >= len(self.images_list)):
            return self.__getitem__(index % len(self.images_list))

        if self.download_dataset:
            # Find the directory that contains the image
            current_dir = self.images_list[index].split(os.path.sep)[-2]

            # Scan the full_path to find all the downloaded directories
            self.downloaded_dirs = scan_path(self.base_path)
            # Remove all the directories that are not the current_dir.
            current_dir_downloaded = False
            for i in range(len(self.downloaded_dirs)):
                if os.path.basename(self.downloaded_dirs[i]) != current_dir:
                    if self.remove_unused_dataset:
                        print(f"Removing {self.downloaded_dirs[i]}")
                        shutil.rmtree(self.downloaded_dirs[i])
                else:
                    current_dir_downloaded = True

            # If current_dir is not downloaded, download it
            if not current_dir_downloaded:
                self.download_and_extract(current_dir, is_dir=True)

        # From here is the same if the dataset was downloaded or not

        try:
            im = Image.open(self.images_list[index]).convert('RGB')
        except Exception as e: # If the image is corrupted, delete it and return the next one
            print(" Exception caught: skipping dataset element.")
            print(f"Error reading image {self.images_list[index]}: {e}")
            del self.images_list[index]
            return self.__getitem__(index)
        
        # Apply the transformations
        if self.transform is not None:
            im = self.transform(im)

        # Same strategy as before if target is corrupted
        try:
            #aquire the target from file
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
    
    #return_fig: if True will return the fig object
    def show_bounding(self, index, classes_to_show=None, return_fig=False): 
        image, targets = self[index]
        self.label, self.bounding = targets["labels"], targets["boxes"]

        fig, ax = plt.subplots(1)
        ax.imshow(image.permute(1, 2, 0))
        # ax.imshow(image)

        for l, bbox in zip(self.label, self.bounding):
            if classes_to_show is not None and self.str_label[l] not in classes_to_show:
                continue
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            x = bbox[0]
            y = bbox[1]

            # Create a rectangle patch
            rect = patches.Rectangle(
                (x, y), width, height, linewidth=1, edgecolor=self.colors[l], facecolor="none")

            # Add the rectangle to the axes
            ax.add_patch(rect)
            ax.text(x, y - 10, self.str_label[l], color=self.colors[l])
            ax.axis('off')

        ax.set_title("Ground Truth")
        if return_fig:
            return fig
        else:
            # Show the plot
            plt.show(block=False)
    
        
