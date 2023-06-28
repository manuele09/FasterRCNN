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

# dataset.show_bounding(0)

# drop_last=True
# pin_memory=True
# shuffle=True
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS, collate_fn=collate_fn)

if False:
    resume_epoch = 0
    resume_batch = 4  # Non è il batch n, ma il batch n+1
    
    dataset.skip_items(resume_batch*BATCH_SIZE + 1)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS, collate_fn=collate_fn)
    load_dict = {"load_from_wandb": False,
                "epoch": 0,
                "batch": 4,
                "path": None}
else:
    load_dict = None
model = FasterModel(dataloader, base_path, load_dict=load_dict)

model.train(print_freq=1, save_freq=1)

# model.evaluate(dataloader)


# Epoch: [0]  [    0/18129]  eta: 15:31:52  lr: 0.000002  loss: 4.8793 (4.8793)  loss_classifier: 3.5329 (3.5329)  loss_box_reg: 0.7846 (0.7846)  loss_objectness: 0.4764 (0.4764)  loss_rpn_box_reg: 0.0853 (0.0853)  time: 3.0842  data: 0.3474  max mem: 7553
# Epoch: [0]  [    1/18129]  eta: 10:29:18  lr: 0.000003  loss: 4.8793 (4.9508)  loss_classifier: 3.5329 (3.5809)  loss_box_reg: 0.7846 (0.8021)  loss_objectness: 0.4764 (0.4791)  loss_rpn_box_reg: 0.0853 (0.0887)  time: 2.0829  data: 0.1884  max mem: 7890
# Epoch: [0]  [    2/18129]  eta: 8:47:32  lr: 0.000004  loss: 5.0223 (4.9862)  loss_classifier: 3.6288 (3.6251)  loss_box_reg: 0.7846 (0.7945)  loss_objectness: 0.4803 (0.4795)  loss_rpn_box_reg: 0.0853 (0.0871)  time: 1.7461  data: 0.1374  max mem: 7890
# Epoch: [0]  [    3/18129]  eta: 7:56:43  lr: 0.000005  loss: 4.9333 (4.9730)  loss_classifier: 3.5683 (3.6109)  loss_box_reg: 0.7846 (0.8011)  loss_objectness: 0.4764 (0.4742)  loss_rpn_box_reg: 0.0853 (0.0868)  time: 1.5780  data: 0.1106  max mem: 7890
# Epoch: [0]  [    4/18129]  eta: 7:26:08  lr: 0.000006  loss: 5.0223 (4.9960)  loss_classifier: 3.5683 (3.6008)  loss_box_reg: 0.8196 (0.8123)  loss_objectness: 0.4803 (0.4873)  loss_rpn_box_reg: 0.0860 (0.0957)  time: 1.4769  data: 0.0945  max mem: 7890
# Epoch: [0]  [    5/18129]  eta: 7:09:23  lr: 0.000007  loss: 5.0223 (5.0448)  loss_classifier: 3.5683 (3.6017)  loss_box_reg: 0.8196 (0.8151)  loss_objectness: 0.4803 (0.5181)  loss_rpn_box_reg: 0.0860 (0.1098)  time: 1.4215  data: 0.0837  max mem: 7890
# Epoch: [0]  [    6/18129]  eta: 6:58:12  lr: 0.000008  loss: 5.0223 (5.0301)  loss_classifier: 3.5683 (3.5896)  loss_box_reg: 0.8196 (0.8138)  loss_objectness: 0.4818 (0.5190)  loss_rpn_box_reg: 0.0922 (0.1076)  time: 1.3846  data: 0.0794  max mem: 7890
# Epoch: [0]  [    7/18129]  eta: 6:50:22  lr: 0.000009  loss: 4.9417 (4.7807)  loss_classifier: 3.5603 (3.4138)  loss_box_reg: 0.8059 (0.7753)  loss_objectness: 0.4803 (0.4841)  loss_rpn_box_reg: 0.0922 (0.1075)  time: 1.3587  data: 0.0758  max mem: 7890
# Epoch: [0]  [    8/18129]  eta: 6:44:02  lr: 0.000010  loss: 5.0223 (4.8362)  loss_classifier: 3.5683 (3.4322)  loss_box_reg: 0.8196 (0.7822)  loss_objectness: 0.4818 (0.5127)  loss_rpn_box_reg: 0.0946 (0.1091)  time: 1.3378  data: 0.0726  max mem: 7890
# Epoch: [0]  [    9/18129]  eta: 6:36:34  lr: 0.000011  loss: 4.9449 (4.8471)  loss_classifier: 3.5603 (3.4423)  loss_box_reg: 0.8119 (0.7852)  loss_objectness: 0.4803 (0.5088)  loss_rpn_box_reg: 0.0946 (0.1108)  time: 1.3132  data: 0.0702  max mem: 7890
# Epoch: [0]  [   10/18129]  eta: 6:30:26  lr: 0.000012  loss: 4.9449 (4.8402)  loss_classifier: 3.5603 (3.4355)  loss_box_reg: 0.8196 (0.7940)  loss_objectness: 0.4803 (0.4992)  loss_rpn_box_reg: 0.1064 (0.1116)  time: 1.2929  data: 0.0669  max mem: 7890
# Epoch: [0]  [   11/18129]  eta: 6:24:58  lr: 0.000013  loss: 4.9417 (4.8037)  loss_classifier: 3.5334 (3.4113)  loss_box_reg: 0.8119 (0.7905)  loss_objectness: 0.4764 (0.4846)  loss_rpn_box_reg: 0.1064 (0.1172)  time: 1.2749  data: 0.0639  max mem: 7890
# Epoch: [0]  [   12/18129]  eta: 6:20:35  lr: 0.000014  loss: 4.9417 (4.8118)  loss_classifier: 3.5334 (3.4133)  loss_box_reg: 0.8196 (0.7967)  loss_objectness: 0.4764 (0.4824)  loss_rpn_box_reg: 0.1195 (0.1195)  time: 1.2604  data: 0.0614  max mem: 7891
# Epoch: [0]  [   13/18129]  eta: 6:16:32  lr: 0.000015  loss: 4.9333 (4.8169)  loss_classifier: 3.5329 (3.4171)  loss_box_reg: 0.8196 (0.7985)  loss_objectness: 0.4737 (0.4812)  loss_rpn_box_reg: 0.1195 (0.1201)  time: 1.2471  data: 0.0592  max mem: 7891
# Epoch: [0]  [   14/18129]  eta: 6:13:05  lr: 0.000016  loss: 4.9333 (4.7543)  loss_classifier: 3.5329 (3.3734)  loss_box_reg: 0.8196 (0.7936)  loss_objectness: 0.4737 (0.4705)  loss_rpn_box_reg: 0.1195 (0.1169)  time: 1.2357  data: 0.0575  max mem: 7891
# Epoch: [0]  [   15/18129]  eta: 6:10:12  lr: 0.000017  loss: 4.9095 (4.7274)  loss_classifier: 3.5169 (3.3577)  loss_box_reg: 0.8196 (0.7991)  loss_objectness: 0.4654 (0.4557)  loss_rpn_box_reg: 0.1064 (0.1149)  time: 1.2262  data: 0.0560  max mem: 7891
# Epoch: [0]  [   16/18129]  eta: 6:07:40  lr: 0.000018  loss: 4.9095 (4.7130)  loss_classifier: 3.5169 (3.3484)  loss_box_reg: 0.8206 (0.8026)  loss_objectness: 0.4654 (0.4483)  loss_rpn_box_reg: 0.1064 (0.1137)  time: 1.2180  data: 0.0545  max mem: 7891
# Epoch: [0]  [   17/18129]  eta: 6:05:28  lr: 0.000019  loss: 4.8829 (4.7027)  loss_classifier: 3.4669 (3.3406)  loss_box_reg: 0.8206 (0.8048)  loss_objectness: 0.4584 (0.4420)  loss_rpn_box_reg: 0.1064 (0.1153)  time: 1.2107  data: 0.0533  max mem: 7891
# Epoch: [0]  [   18/18129]  eta: 6:03:36  lr: 0.000020  loss: 4.8829 (4.6875)  loss_classifier: 3.4669 (3.3310)  loss_box_reg: 0.8225 (0.8061)  loss_objectness: 0.4584 (0.4352)  loss_rpn_box_reg: 0.1120 (0.1152)  time: 1.2046  data: 0.0521  max mem: 7891
# Epoch: [0]  [   19/18129]  eta: 6:03:00  lr: 0.000021  loss: 4.8793 (4.6659)  loss_classifier: 3.4371 (3.3162)  loss_box_reg: 0.8206 (0.8057)  loss_objectness: 0.4558 (0.4275)  loss_rpn_box_reg: 0.1120 (0.1164)  time: 1.2027  data: 0.0525  max mem: 7891
# Epoch: [0]  [   20/18129]  eta: 6:02:43  lr: 0.000022  loss: 4.7716 (4.6470)  loss_classifier: 3.3668 (3.3011)  loss_box_reg: 0.8225 (0.8072)  loss_objectness: 0.4027 (0.4221)  loss_rpn_box_reg: 0.1195 (0.1166)  time: 1.1077  data: 0.0375  max mem: 7891
# Epoch: [0]  [   21/18129]  eta: 6:01:37  lr: 0.000023  loss: 4.5282 (4.6265)  loss_classifier: 3.2070 (3.2828)  loss_box_reg: 0.8246 (0.8080)  loss_objectness: 0.3783 (0.4201)  loss_rpn_box_reg: 0.1195 (0.1156)  time: 1.1097  data: 0.0375  max mem: 7891
# Epoch: [0]  [   22/18129]  eta: 6:02:42  lr: 0.000024  loss: 4.4827 (4.5992)  loss_classifier: 3.2006 (3.2618)  loss_box_reg: 0.8246 (0.8081)  loss_objectness: 0.3363 (0.4140)  loss_rpn_box_reg: 0.1195 (0.1153)  time: 1.1203  data: 0.0375  max mem: 7891
# Epoch: [0]  [   23/18129]  eta: 6:03:41  lr: 0.000025  loss: 4.4131 (4.5747)  loss_classifier: 3.1587 (3.2442)  loss_box_reg: 0.8268 (0.8089)  loss_objectness: 0.3299 (0.4068)  loss_rpn_box_reg: 0.1195 (0.1150)  time: 1.1306  data: 0.0377  max mem: 7891
# Epoch: [0]  [   24/18129]  eta: 6:03:14  lr: 0.000026  loss: 4.4016 (4.5540)  loss_classifier: 3.1453 (3.2260)  loss_box_reg: 0.8268 (0.8105)  loss_objectness: 0.3249 (0.4024)  loss_rpn_box_reg: 0.1190 (0.1151)  time: 1.1355  data: 0.0412  max mem: 7891
# Epoch: [0]  [   25/18129]  eta: 6:02:54  lr: 0.000027  loss: 4.3231 (4.5194)  loss_classifier: 3.1222 (3.2033)  loss_box_reg: 0.8246 (0.8109)  loss_objectness: 0.3207 (0.3921)  loss_rpn_box_reg: 0.1120 (0.1132)  time: 1.1371  data: 0.0421  max mem: 7891


# Epoch: [0]  [    0/18129]  eta: 15:52:16  lr: 0.000002  loss: 4.8848 (4.8848)  loss_classifier: 3.5425 (3.5425)  loss_box_reg: 0.7870 (0.7870)  loss_objectness: 0.4700 (0.4700)  loss_rpn_box_reg: 0.0853 (0.0853)  time: 3.1516  data: 0.3356  max mem: 7553
# Epoch: [0]  [    1/18129]  eta: 10:51:54  lr: 0.000003  loss: 4.8848 (4.9703)  loss_classifier: 3.5425 (3.6054)  loss_box_reg: 0.7870 (0.8003)  loss_objectness: 0.4700 (0.4759)  loss_rpn_box_reg: 0.0853 (0.0887)  time: 2.1577  data: 0.1909  max mem: 7890
# Epoch: [0]  [    2/18129]  eta: 9:09:43  lr: 0.000004  loss: 5.0548 (4.9985)  loss_classifier: 3.6683 (3.6404)  loss_box_reg: 0.7870 (0.7940)  loss_objectness: 0.4790 (0.4769)  loss_rpn_box_reg: 0.0853 (0.0871)  time: 1.8196  data: 0.1393  max mem: 7890
# Epoch: [0]  [    3/18129]  eta: 8:19:47  lr: 0.000005  loss: 4.9501 (4.9864)  loss_classifier: 3.5843 (3.6264)  loss_box_reg: 0.7870 (0.7998)  loss_objectness: 0.4700 (0.4734)  loss_rpn_box_reg: 0.0853 (0.0868)  time: 1.6544  data: 0.1131  max mem: 7890
# Epoch: [0]  [    4/18129]  eta: 7:48:07  lr: 0.000006  loss: 5.0548 (5.0117)  loss_classifier: 3.5843 (3.6152)  loss_box_reg: 0.8136 (0.8139)  loss_objectness: 0.4790 (0.4870)  loss_rpn_box_reg: 0.0860 (0.0957)  time: 1.5497  data: 0.0969  max mem: 7890
# Epoch: [0]  [    5/18129]  eta: 7:28:13  lr: 0.000007  loss: 5.0548 (5.0563)  loss_classifier: 3.5843 (3.6116)  loss_box_reg: 0.8136 (0.8179)  loss_objectness: 0.4790 (0.5170)  loss_rpn_box_reg: 0.0860 (0.1098)  time: 1.4839  data: 0.0869  max mem: 7890
# Epoch: [0]  [    6/18129]  eta: 7:11:11  lr: 0.000008  loss: 5.0548 (5.0397)  loss_classifier: 3.5843 (3.6000)  loss_box_reg: 0.8136 (0.8142)  loss_objectness: 0.4818 (0.5179)  loss_rpn_box_reg: 0.0922 (0.1076)  time: 1.4276  data: 0.0791  max mem: 7890
# Epoch: [0]  [    7/18129]  eta: 6:58:23  lr: 0.000009  loss: 4.9501 (4.7905)  loss_classifier: 3.5702 (3.4232)  loss_box_reg: 0.7920 (0.7755)  loss_objectness: 0.4790 (0.4842)  loss_rpn_box_reg: 0.0922 (0.1075)  time: 1.3853  data: 0.0738  max mem: 7890
# Epoch: [0]  [    8/18129]  eta: 6:48:44  lr: 0.000010  loss: 5.0548 (4.8474)  loss_classifier: 3.5843 (3.4426)  loss_box_reg: 0.8136 (0.7830)  loss_objectness: 0.4818 (0.5127)  loss_rpn_box_reg: 0.0946 (0.1091)  time: 1.3534  data: 0.0695  max mem: 7890
# Epoch: [0]  [    9/18129]  eta: 6:40:47  lr: 0.000011  loss: 4.9501 (4.8567)  loss_classifier: 3.5702 (3.4514)  loss_box_reg: 0.8061 (0.7853)  loss_objectness: 0.4790 (0.5093)  loss_rpn_box_reg: 0.0946 (0.1108)  time: 1.3271  data: 0.0658  max mem: 7890
# Epoch: [0]  [   10/18129]  eta: 6:34:15  lr: 0.000012  loss: 4.9501 (4.8509)  loss_classifier: 3.5702 (3.4451)  loss_box_reg: 0.8136 (0.7944)  loss_objectness: 0.4790 (0.4998)  loss_rpn_box_reg: 0.1064 (0.1116)  time: 1.3055  data: 0.0625  max mem: 7890
# Epoch: [0]  [   11/18129]  eta: 6:28:54  lr: 0.000013  loss: 4.9409 (4.8133)  loss_classifier: 3.5425 (3.4199)  loss_box_reg: 0.8061 (0.7909)  loss_objectness: 0.4785 (0.4852)  loss_rpn_box_reg: 0.1064 (0.1172)  time: 1.2879  data: 0.0602  max mem: 7890
# Epoch: [0]  [   12/18129]  eta: 6:24:14  lr: 0.000014  loss: 4.9409 (4.8207)  loss_classifier: 3.5425 (3.4218)  loss_box_reg: 0.8136 (0.7959)  loss_objectness: 0.4785 (0.4834)  loss_rpn_box_reg: 0.1195 (0.1195)  time: 1.2725  data: 0.0580  max mem: 7891
# Epoch: [0]  [   13/18129]  eta: 6:21:37  lr: 0.000015  loss: 4.9404 (4.8259)  loss_classifier: 3.5305 (3.4257)  loss_box_reg: 0.8136 (0.7979)  loss_objectness: 0.4700 (0.4822)  loss_rpn_box_reg: 0.1195 (0.1201)  time: 1.2640  data: 0.0562  max mem: 7891
# Epoch: [0]  [   14/18129]  eta: 6:18:17  lr: 0.000016  loss: 4.9404 (4.7617)  loss_classifier: 3.5305 (3.3805)  loss_box_reg: 0.8136 (0.7928)  loss_objectness: 0.4700 (0.4714)  loss_rpn_box_reg: 0.1195 (0.1169)  time: 1.2530  data: 0.0555  max mem: 7891
# Epoch: [0]  [   15/18129]  eta: 6:16:29  lr: 0.000017  loss: 4.9089 (4.7340)  loss_classifier: 3.5304 (3.3639)  loss_box_reg: 0.8136 (0.7990)  loss_objectness: 0.4667 (0.4562)  loss_rpn_box_reg: 0.1064 (0.1149)  time: 1.2471  data: 0.0552  max mem: 7891
# Epoch: [0]  [   16/18129]  eta: 6:14:48  lr: 0.000018  loss: 4.9089 (4.7219)  loss_classifier: 3.5304 (3.3560)  loss_box_reg: 0.8170 (0.8033)  loss_objectness: 0.4667 (0.4489)  loss_rpn_box_reg: 0.1064 (0.1137)  time: 1.2416  data: 0.0548  max mem: 7891
# Epoch: [0]  [   17/18129]  eta: 6:13:36  lr: 0.000019  loss: 4.8933 (4.7114)  loss_classifier: 3.4758 (3.3470)  loss_box_reg: 0.8170 (0.8060)  loss_objectness: 0.4629 (0.4430)  loss_rpn_box_reg: 0.1064 (0.1153)  time: 1.2376  data: 0.0550  max mem: 7891
# Epoch: [0]  [   18/18129]  eta: 6:11:51  lr: 0.000020  loss: 4.8933 (4.6955)  loss_classifier: 3.4758 (3.3369)  loss_box_reg: 0.8228 (0.8070)  loss_objectness: 0.4629 (0.4364)  loss_rpn_box_reg: 0.1120 (0.1152)  time: 1.2319  data: 0.0549  max mem: 7891
# Epoch: [0]  [   19/18129]  eta: 6:09:46  lr: 0.000021  loss: 4.8848 (4.6742)  loss_classifier: 3.4447 (3.3233)  loss_box_reg: 0.8170 (0.8056)  loss_objectness: 0.4616 (0.4289)  loss_rpn_box_reg: 0.1120 (0.1164)  time: 1.2251  data: 0.0538  max mem: 7891
# Epoch: [0]  [   20/18129]  eta: 6:07:57  lr: 0.000022  loss: 4.7926 (4.6551)  loss_classifier: 3.3821 (3.3076)  loss_box_reg: 0.8228 (0.8069)  loss_objectness: 0.4054 (0.4239)  loss_rpn_box_reg: 0.1195 (0.1166)  time: 1.1225  data: 0.0390  max mem: 7891
# Epoch: [0]  [   21/18129]  eta: 6:06:16  lr: 0.000023  loss: 4.5319 (4.6348)  loss_classifier: 3.2303 (3.2897)  loss_box_reg: 0.8228 (0.8076)  loss_objectness: 0.3797 (0.4219)  loss_rpn_box_reg: 0.1195 (0.1157)  time: 1.1192  data: 0.0384  max mem: 7891
# Epoch: [0]  [   22/18129]  eta: 6:04:50  lr: 0.000024  loss: 4.5286 (4.6071)  loss_classifier: 3.1938 (3.2681)  loss_box_reg: 0.8228 (0.8080)  loss_objectness: 0.3432 (0.4157)  loss_rpn_box_reg: 0.1195 (0.1153)  time: 1.1174  data: 0.0384  max mem: 7891
# Epoch: [0]  [   23/18129]  eta: 6:03:53  lr: 0.000025  loss: 4.4099 (4.5823)  loss_classifier: 3.1560 (3.2501)  loss_box_reg: 0.8240 (0.8093)  loss_objectness: 0.3314 (0.4080)  loss_rpn_box_reg: 0.1195 (0.1150)  time: 1.1161  data: 0.0388  max mem: 7891
# Epoch: [0]  [   24/18129]  eta: 6:03:36  lr: 0.000026  loss: 4.3997 (4.5601)  loss_classifier: 3.1434 (3.2308)  loss_box_reg: 0.8240 (0.8106)  loss_objectness: 0.3257 (0.4036)  loss_rpn_box_reg: 0.1191 (0.1151)  time: 1.1188  data: 0.0397  max mem: 7891
# Epoch: [0]  [   25/18129]  eta: 6:03:17  lr: 0.000027  loss: 4.3192 (4.5252)  loss_classifier: 3.1135 (3.2078)  loss_box_reg: 0.8228 (0.8110)  loss_objectness: 0.3245 (0.3933)  loss_rpn_box_reg: 0.1120 (0.1132)  time: 1.1201  data: 0.0407  max mem: 7891

# Epoch: [0]  [    0/18129]  eta: 17:04:25  lr: 0.000002  loss: 4.9181 (4.9181)  loss_classifier: 3.5649 (3.5649)  loss_box_reg: 0.7932 (0.7932)  loss_objectness: 0.4746 (0.4746)  loss_rpn_box_reg: 0.0853 (0.0853)  time: 3.3904  data: 0.5258  max mem: 7552
# Epoch: [0]  [    1/18129]  eta: 11:17:44  lr: 0.000003  loss: 4.9181 (4.9853)  loss_classifier: 3.5649 (3.6096)  loss_box_reg: 0.7932 (0.8059)  loss_objectness: 0.4746 (0.4811)  loss_rpn_box_reg: 0.0853 (0.0887)  time: 2.2432  data: 0.2785  max mem: 7889
# Epoch: [0]  [    2/18129]  eta: 9:31:26  lr: 0.000004  loss: 5.0357 (5.0021)  loss_classifier: 3.6543 (3.6350)  loss_box_reg: 0.7932 (0.7992)  loss_objectness: 0.4805 (0.4809)  loss_rpn_box_reg: 0.0853 (0.0871)  time: 1.8915  data: 0.2103  max mem: 7889
# Epoch: [0]  [    3/18129]  eta: 8:31:54  lr: 0.000005  loss: 4.9584 (4.9912)  loss_classifier: 3.5874 (3.6231)  loss_box_reg: 0.7932 (0.8053)  loss_objectness: 0.4746 (0.4761)  loss_rpn_box_reg: 0.0853 (0.0868)  time: 1.6945  data: 0.1610  max mem: 7889
# Epoch: [0]  [    4/18129]  eta: 7:58:12  lr: 0.000006  loss: 5.0357 (5.0179)  loss_classifier: 3.5874 (3.6146)  loss_box_reg: 0.8185 (0.8179)  loss_objectness: 0.4805 (0.4897)  loss_rpn_box_reg: 0.0860 (0.0957)  time: 1.5830  data: 0.1394  max mem: 7890
# Epoch: [0]  [    5/18129]  eta: 7:34:46  lr: 0.000007  loss: 5.0357 (5.0612)  loss_classifier: 3.5874 (3.6114)  loss_box_reg: 0.8185 (0.8218)  loss_objectness: 0.4805 (0.5182)  loss_rpn_box_reg: 0.0860 (0.1098)  time: 1.5055  data: 0.1214  max mem: 7890
# Epoch: [0]  [    6/18129]  eta: 7:17:41  lr: 0.000008  loss: 5.0357 (5.0428)  loss_classifier: 3.5874 (3.5961)  loss_box_reg: 0.8185 (0.8196)  loss_objectness: 0.4877 (0.5194)  loss_rpn_box_reg: 0.0921 (0.1076)  time: 1.4491  data: 0.1086  max mem: 7890
# Epoch: [0]  [    7/18129]  eta: 7:04:11  lr: 0.000009  loss: 4.9584 (4.7924)  loss_classifier: 3.5808 (3.4190)  loss_box_reg: 0.8062 (0.7810)  loss_objectness: 0.4805 (0.4849)  loss_rpn_box_reg: 0.0921 (0.1075)  time: 1.4044  data: 0.0990  max mem: 7890
# Epoch: [0]  [    8/18129]  eta: 6:53:49  lr: 0.000010  loss: 5.0357 (4.8483)  loss_classifier: 3.5874 (3.4388)  loss_box_reg: 0.8185 (0.7865)  loss_objectness: 0.4877 (0.5139)  loss_rpn_box_reg: 0.0946 (0.1091)  time: 1.3702  data: 0.0918  max mem: 7890
# Epoch: [0]  [    9/18129]  eta: 6:45:26  lr: 0.000011  loss: 4.9584 (4.8575)  loss_classifier: 3.5808 (3.4484)  loss_box_reg: 0.8062 (0.7882)  loss_objectness: 0.4805 (0.5101)  loss_rpn_box_reg: 0.0946 (0.1108)  time: 1.3425  data: 0.0858  max mem: 7890
# Epoch: [0]  [   10/18129]  eta: 6:38:51  lr: 0.000012  loss: 4.9584 (4.8520)  loss_classifier: 3.5808 (3.4440)  loss_box_reg: 0.8185 (0.7961)  loss_objectness: 0.4805 (0.5003)  loss_rpn_box_reg: 0.1064 (0.1116)  time: 1.3208  data: 0.0813  max mem: 7891
# Epoch: [0]  [   11/18129]  eta: 6:33:03  lr: 0.000013  loss: 4.9398 (4.8158)  loss_classifier: 3.5649 (3.4197)  loss_box_reg: 0.8062 (0.7925)  loss_objectness: 0.4752 (0.4863)  loss_rpn_box_reg: 0.1064 (0.1173)  time: 1.3017  data: 0.0774  max mem: 7891
# Epoch: [0]  [   12/18129]  eta: 6:28:15  lr: 0.000014  loss: 4.9398 (4.8231)  loss_classifier: 3.5649 (3.4222)  loss_box_reg: 0.8185 (0.7977)  loss_objectness: 0.4752 (0.4838)  loss_rpn_box_reg: 0.1196 (0.1195)  time: 1.2858  data: 0.0740  max mem: 7891
# Epoch: [0]  [   13/18129]  eta: 6:24:02  lr: 0.000015  loss: 4.9320 (4.8282)  loss_classifier: 3.5349 (3.4260)  loss_box_reg: 0.8185 (0.7996)  loss_objectness: 0.4746 (0.4824)  loss_rpn_box_reg: 0.1196 (0.1201)  time: 1.2719  data: 0.0711  max mem: 7891
# Epoch: [0]  [   14/18129]  eta: 6:20:23  lr: 0.000016  loss: 4.9320 (4.7640)  loss_classifier: 3.5349 (3.3810)  loss_box_reg: 0.8185 (0.7945)  loss_objectness: 0.4746 (0.4715)  loss_rpn_box_reg: 0.1196 (0.1169)  time: 1.2599  data: 0.0688  max mem: 7891
# Epoch: [0]  [   15/18129]  eta: 6:17:59  lr: 0.000017  loss: 4.9181 (4.7371)  loss_classifier: 3.5046 (3.3656)  loss_box_reg: 0.8185 (0.8002)  loss_objectness: 0.4649 (0.4563)  loss_rpn_box_reg: 0.1064 (0.1149)  time: 1.2520  data: 0.0666  max mem: 7892
# Epoch: [0]  [   16/18129]  eta: 6:22:06  lr: 0.000018  loss: 4.9181 (4.7236)  loss_classifier: 3.5046 (3.3568)  loss_box_reg: 0.8236 (0.8041)  loss_objectness: 0.4649 (0.4490)  loss_rpn_box_reg: 0.1064 (0.1137)  time: 1.2657  data: 0.0697  max mem: 7892
# Epoch: [0]  [   17/18129]  eta: 6:20:26  lr: 0.000019  loss: 4.9112 (4.7137)  loss_classifier: 3.4756 (3.3494)  loss_box_reg: 0.8236 (0.8059)  loss_objectness: 0.4615 (0.4430)  loss_rpn_box_reg: 0.1064 (0.1153)  time: 1.2603  data: 0.0689  max mem: 7892
# Epoch: [0]  [   18/18129]  eta: 6:18:40  lr: 0.000020  loss: 4.9112 (4.6961)  loss_classifier: 3.4756 (3.3384)  loss_box_reg: 0.8236 (0.8065)  loss_objectness: 0.4615 (0.4360)  loss_rpn_box_reg: 0.1120 (0.1152)  time: 1.2545  data: 0.0671  max mem: 7892
# Epoch: [0]  [   19/18129]  eta: 6:17:27  lr: 0.000021  loss: 4.8936 (4.6749)  loss_classifier: 3.4518 (3.3242)  loss_box_reg: 0.8185 (0.8064)  loss_objectness: 0.4538 (0.4278)  loss_rpn_box_reg: 0.1120 (0.1164)  time: 1.2506  data: 0.0667  max mem: 7892
# Epoch: [0]  [   20/18129]  eta: 6:15:15  lr: 0.000022  loss: 4.7977 (4.6553)  loss_classifier: 3.4005 (3.3082)  loss_box_reg: 0.8236 (0.8081)  loss_objectness: 0.4032 (0.4224)  loss_rpn_box_reg: 0.1196 (0.1166)  time: 1.1360  data: 0.0422  max mem: 7892
# Epoch: [0]  [   21/18129]  eta: 6:13:10  lr: 0.000023  loss: 4.5456 (4.6344)  loss_classifier: 3.2246 (3.2902)  loss_box_reg: 0.8236 (0.8084)  loss_objectness: 0.3743 (0.4202)  loss_rpn_box_reg: 0.1196 (0.1157)  time: 1.1358  data: 0.0422  max mem: 7892
# Epoch: [0]  [   22/18129]  eta: 6:11:17  lr: 0.000024  loss: 4.5084 (4.6061)  loss_classifier: 3.2154 (3.2687)  loss_box_reg: 0.8236 (0.8080)  loss_objectness: 0.3411 (0.4140)  loss_rpn_box_reg: 0.1196 (0.1153)  time: 1.1311  data: 0.0402  max mem: 7892
# Epoch: [0]  [   23/18129]  eta: 6:09:35  lr: 0.000025  loss: 4.4171 (4.5815)  loss_classifier: 3.1522 (3.2521)  loss_box_reg: 0.8175 (0.8080)  loss_objectness: 0.3317 (0.4065)  loss_rpn_box_reg: 0.1196 (0.1150)  time: 1.1308  data: 0.0414  max mem: 7892
# Epoch: [0]  [   24/18129]  eta: 6:08:05  lr: 0.000026  loss: 4.3796 (4.5592)  loss_classifier: 3.1405 (3.2322)  loss_box_reg: 0.8175 (0.8095)  loss_objectness: 0.3316 (0.4023)  loss_rpn_box_reg: 0.1191 (0.1151)  time: 1.1290  data: 0.0405  max mem: 7892
# Epoch: [0]  [   25/18129]  eta: 6:06:35  lr: 0.000027  loss: 4.3332 (4.5256)  loss_classifier: 3.1346 (3.2103)  loss_box_reg: 0.8175 (0.8100)  loss_objectness: 0.3191 (0.3921)  loss_rpn_box_reg: 0.1120 (0.1132)  time: 1.1277  data: 0.0406  max mem: 7892
