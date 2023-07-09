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
import sys
import threading
import time

from custom_utils import change_extension, from_path_to_names

# takes as input a list_file, containing the paths of the images.
# modify the paths in the list_file, in order to mantain n_dirs_to_mantain directories
# and adding new_root_path as root.


def modify_list(list_file, n_dirs_to_mantain, new_root_path):
    # Example of name in input_file: /dataset_1088x612/27_03_19_19_15_32/000/000001.png
    # n_dirst_to_mantain: 2
    # new_root_path: /Root
    # Output name: /Root/27_03_19_19_15_32/000/000001.png
    new_list = []
    with open(list_file, "r") as f:
        file_paths = f.readlines()
        for i in range(len(file_paths)):
            dirs, file_name = os.path.split(
                file_paths[i].replace('\\', os.sep))
            for j in range(n_dirs_to_mantain):
                dirs, d = os.path.split(dirs)
                file_name = os.path.join(d, file_name)
            file_name = os.path.join(new_root_path, file_name)
            new_list.append(file_name.strip())
    return new_list


def find_path(name, path):
    for root, dirs, files in os.walk(path):
        if name in dirs or name in files:
            return os.path.join(root, name)
    return None


def scan_path(path):
    dirs = [os.path.join(path, name) for name in os.listdir(
        path) if os.path.isdir(os.path.join(path, name))]
    return dirs


class RealDataset(Dataset):

    # return a dictionary with the following keys: "image", "target".
    def read_target_from_file(self, index, im_width, im_height):
        # bounding file format: x_center, y_center, width, height (all expressed in percentages)
        # bounding new format: x_min, y_min, x_max, y_max (expressed in the real image range)
        labels = []
        bounding_boxes = []

        target_path = change_extension(self.images_list[index], ".txt")
        with open(target_path, 'r') as file:
            for line in file:
                # label     x_center  y_center  width     height
                target_str = line.strip().split()
                # target[0] target[1] target[2] target[3] target[4]
                target = [float(x) for x in target_str]

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

    # images_root: path containg all the images
    # image_list_path: path of the file containing the list of all the images name
    def __init__(self, base_path, images_list=None, list_file_name=None,  transform=None, download_virtual_dataset=False):
        self.base_path = base_path
        self.download_virtual_dataset = download_virtual_dataset
        self.images_list = images_list
        self.transform = transform
        
        self.str_label = ["No Elmet", "Elmet", "Welding Mask",
                          "Ear Protection", "No Gilet", "Gilet", "Person"]
        self.colors = ['red', 'blue', 'green',
                       'orange', 'purple', 'pink', "brown"]
        # self.dirs_ids = {"train.virtual.txt": "ESRgAfYQkchGj4Hjfl_lZLMBoLNTrhkHwPJzYBGsrt4SeA",
        #                  "valid.virtual.txt": "EXRzg_URR-ZEoYMpYH6W8R4BJWUVq4HMKZe-bvoq8ngUXw",
        #                  "27_03_19_19_15_32": "EVujRKjyKSJDiQ_8b-46r7sBSoY7yMre_UiHVXy4W3c14w",
        #                  "27_03_19_19_47_44": "EVTdTdDHT6FPkTAh-zK2JaoBQFNXpsHJfiKtOlxlga5dDQ",
        #                  "27_03_19_20_16_23":  "EZnnM9VW7qxCpuOoMd3DD70BTxf3qTUSzSFo1ItAcpzvVQ",
        #                  "27_03_19_20_43_51": "EbFMsp8MSwlDgXkS0EguZkwBhw5DCgi2nO3yTtjl426WMQ",
        #                  "27_03_19_21_09_00": "EUTcg4dh9G1Ji1QSQ34MTIIBBavjnOQYRAA0RsEx_Z4VqA",
        #                  "29_03_19_03_39_42": "ESDmLETIsShNlE2oEJI1D2QB2trOJCjWsJsc3O-dssu_ag",
        #                  "29_03_19_04_06_35": "EdoGdVHUsLRJiMxh0a1VBBEBrYeC2eX0Elvs1Jhq_b2gmw",
        #                  "29_03_19_04_34_54": "Ed4ZNXeY9a9HgY4T6MJJ7f0ByGK1EwbQmTxI8m6ijxDiBQ",
        #                  "29_03_19_05_01_34": "Ee9SYf6BuCxJiCpNbdXMTxoBEVHGA66aSHfsdmS_leHswQ",
        #                  "29_03_19_05_27_16": "Ed5KYH5YpX9BuS-5L4XDI9UBslYiXpRFs_UR8G_lBeQZzA",
        #                  "29_03_19_05_51_06": "EUi2CAMAvChJhPgQ0WNZGSoBGkuF0RQtD-4JXkhdJgNrEg",
        #                  "29_03_19_06_18_10": "EVnFqP7dx4VDnQ8XhiCa3W8B7VnESrnCYDghjrfGWU6xYQ",
        #                  "29_03_19_06_49_23": "EZI6BtKNLKNPqvXrH1WV-yQBX8KE-aAYfxaYqAZwT2048A",
        #                  "29_03_19_07_14_43": "EcrxUMpbayNLgMFvdH99rcQBCPedcn6QKavKeecMAPOGDA",
        #                  "29_03_19_07_45_24": "ERFr_pA4MRRIiemPRdMcoJwBuRjXdg62UYsgm9NAR1dDOA",
        #                  "29_03_19_08_12_28": "EWltsNr9UHdJsiC88YFQdmoB76AwtIFy6wea4oHZMRCNTw",
        #                  "29_03_19_08_39_27": "EaQFe7l04HxMpqzaYsxxQdUBLAtAKfESjI5jHgOg8Yz8PQ",
        #                  "29_03_19_09_05_50": "EZC6nbmQuz5OjFBKaTIoeRMBDSrtW6bNG-HNbB-F8DV3_Q",
        #                  "29_03_19_11_23_30": "EdDkwmpyxRpBqyCRbW3_75ABg4rPucqeMs-3afhEkEE7fA",
        #                  "29_03_19_11_48_52": "ERkF1A2H8NZPrIXx2EIZcyoBw10Q9k2QG2gIzvmMnxUXTA",
        #                  "29_03_19_12_11_24": "Ec51v9AKNTRPo1DI-YFLkrwBZptCy3XcPG-zZXHGHPYwsA",
        #                  "29_03_19_12_36_41": "ET5MAzxCLdRApgu-Yd4QSIsBIUIK2ZO1gc5VCgPpyC2hVw",
        #                  "29_03_19_13_16_22": "Eb6aTsYREItFrJb0kcbIbYEBWNHdlihTP7GTKOdgXC18Hg",
        #                  "29_03_19_13_40_24": "Ec8f5L291nBDkB6Z8XdbuZQB5fzAu9Rvb_3b0j8331ihNg",
        #                  "29_03_19_14_05_16": "EW0WMufbzspGim2ktl1jRosBwgeVU351rGaNTe4uy4VgRw",
        #                  "29_03_19_14_34_53": "EShJyt7_ELVOjxxmje0tYK8BImM7XYIlnLXaBZLm0f5iCQ",
        #                  "29_03_19_15_07_02": "EfjQr--s7u1NrkA13UNBZisBk596wqeFYFAr9qshRuefsw",
        #                  "29_03_19_15_39_29": "EXxy-jkcuLtHpR5vWX2y3ukBctruwvFqO9KNb2PshLSuow",
        #                  "29_03_19_16_06_55": "EQ1oEyDMhs9Au2AtS9wyIU4BnWKFxAg6UJyL8oD8I-8y2w",
        #                  "29_03_19_16_29_52": "EdKf-fzUaxxPk8wE_lykl2wBkec_JR4gjEtXpjdzuIl3Qw"}

        self.dirs_ids = {"train.virtual.txt": "ESRH0_-lm7BKr4jRUZRyhQgBlStl96JNnQ1mpKn2mhXw8g",
                         "valid.virtual.txt": "ESV7PFigkP5EicRmdChtQdMBjBTonHCgEM0-OHhk4i3phQ",
                         "group_0": "ERf726zSyetKpEMBoCgER98BpjOC1DCFxG4JweGMckKRKA",
                         "group_1": "EXJ8NAapu-1OpMtLrZHB2nYBeA4ihaj25MkqP7_X5cU0aQ",
                         "group_2": "EYr2Vm3cTHVPn71sCXVYh-oB4kwfPLf3h92xksdRWSkl5A",
                         "group_3": "ES3TDB2YyT5MrntnsXOb6lYBh9jsnYLrG1eCvsTC5H-_ZA",
                         "group_4": "EQF1h-XDbpFIkbY85z-atTcBahhuLLh6m72qSkwIPqjSNQ",
                         "group_5": "EbNIPDSnqmtGsOgEdZIG3CIBKuctGxA43hPaX4R4EmZP7A",
                         "group_6": "EUHy3XNiFEZFpzjWxKyd6RQBy_iMstYTpId550fw4CBvHA",
                         "group_7": "ERiFzJOn7hRFv8MiEXGRqqcBIkOIpgd9GKC0RRZfQZCUzw",
                         "group_8": "EandlODyoGROn6CxM_luD1UBBmHnCL4QqG3Qz4IGxYffeQ",
                         "group_9": "EROuH231Jg1AudGXiA4bSqUBsATbx7rACNtSHgC-o_rsGQ",
                         "group_10": "EUT3kjz--2xIjA0DM-qe8OIBRAfnbFhBtxw4UWNfL8EElA",
                         "group_11": "ERqlsvhQpYtJup8A-EVL3w0BnU591XnXHAo4yj360K7UGw",
                         "group_12": "EetY8AztGlBFhlDKyj3kTuYB-g_HiDqUh8J5G7i_nsG7cQ",
                         "group_13": "EeoWWxmf8yFMrfyKlz2JrfYBxK0jKp_VkJKb-eyNHPQ43Q",
                         "group_14": "EUj7sdgMZ2hDvuagyX4AYqUBlA0uA8hm0FhUEMGHDIo-ZA",
                         "group_15": "EfuGZjOJKLxOtgs1iJ8vyXABSwRkz8KCdO994NEYv48ypg",
                         "group_16": "Ee2G-cGNgUBFr13PUZTOkKYB07REnEJDK8lC4Cnx_sQljg",
                         "group_17": "EfaA32jOwR5IopCNiP_EH4kBGjUvViNgJ04fls6CS79EcQ",
                         "group_18": "Eecun1vLi-VChu3tOFXjNAoBkTyvlMzieELscP9osdeUUg",
                         "group_19": "ES4WnpNHHTxPrtsQvtHuqj8BKHJqzKJDmBY-In0ehWThDg",
                         "group_20": "EZ2KJoTua-1LvaR74opw2YoB-G6OlNUPJKVFNPsKGp_X_g"
                         }
        if self.download_virtual_dataset:
            if list_file_name is None:
                print("Error: list_file_name is None")
                return

            # create the base_path if not exists, and download the list_file
            if not os.path.exists(self.base_path):
                print("Creating base_path")
                os.makedirs(self.base_path)
                self.download_and_extract(list_file_name, is_dir=False)
            # if the base_path exists but the list_file doesn't, download it
            if find_path(list_file_name, self.base_path) is None:
                self.download_and_extract(list_file_name, is_dir=False)

            # finds the list file path, modifies the relative paths in the list.
            self.images_list = modify_list(
                find_path(list_file_name, self.base_path), 1, self.base_path)

            # it is the path that will contain all the folders of the virtual dataset: 27_03_19_19_15_32, 27_03_19_19_47_44, ...
            self.full_path = os.path.join(
                self.base_path, self.images_list[0].split(os.path.sep)[-4])
            # usually it is base_path + "/dataset_1088x612"

        # Now we should have a complete images_list, that may be generated by the downloaded file
        # or given as input by the user
        if self.images_list is None:
            print("Error: images_list is None")
            return
        self.original_images_list = self.images_list.copy()

    def skip_items(self, start_index):
        self.images_list = self.original_images_list[start_index:]
        print(f"Skipping {start_index} items")

    # Downloads the file_name.zip from the shared link, extracts it in the base_path
    # and deletes the zip file to save space

    def download_and_extract(self, file_name, is_dir):
        # Download the zip
        print(f"Downloading {file_name}.zip ...")
        file_id = self.dirs_ids[file_name]
        out_path = os.path.join(self.base_path, file_name) + ".zip"
        download_url = f"https://studentiunict-my.sharepoint.com/:u:/g/personal/cnnmnl01r09m088w_studium_unict_it/{file_id}?download=1"
        command = f"wget --no-check-certificate -O {out_path} {download_url} > /dev/null 2>&1"
        
        timeout = 200
        while(True):
            try:
                subprocess.run(command, shell=True, timeout=timeout)
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

    def __getitem__(self, index):
        # Circular indexing to avoid index out of range
        if (index >= len(self.images_list)):
            # return None
            return self.__getitem__(index % len(self.images_list))

        if self.download_virtual_dataset:

            # Find the directory that contains the image
            current_dir = self.images_list[index].split(os.path.sep)[-2]

            # Scan the full_path to find all the downloaded directories
            self.downloaded_dirs = scan_path(self.base_path)
            # Remove all the directories that are not the current_dir.
            current_dir_downloaded = False
            for i in range(len(self.downloaded_dirs)):
                if os.path.basename(self.downloaded_dirs[i]) != current_dir:
                    print(f"Removing {self.downloaded_dirs[i]}")
                    # aggiungere opzione per non rimuovere
                    shutil.rmtree(self.downloaded_dirs[i])
                else:
                    current_dir_downloaded = True

            # If current_dir is not downloaded, download it
            if not current_dir_downloaded:
                self.download_and_extract(current_dir, is_dir=True)

        # From here is the same if the dataset was downloaded or not

        try:
            im = Image.open(self.images_list[index]).convert('RGB')
        # If the image is corrupted, delete it and return the next one
        except Exception as e:
            print(" Exception caught: skipping dataset element.")
            print(f"Error reading image {self.images_list[index]}: {e}")
            del self.images_list[index]
            return self.__getitem__(index)
        
        # Apply the transformations
        if self.transform is not None:
            im = self.transform(im)


        # Same strategy as before if target is corrupted
        try:
            target = self.read_target_from_file(
                index, im.shape[2], im.shape[1])
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

    def show_bounding(self, index, classes_to_show=None): 
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

        # Show the plot
        plt.show()
        return fig
    
        image, targets = self[index]
