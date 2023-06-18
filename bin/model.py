import torch
import torchvision
import datetime
from custom_utils import *
from torch.utils.tensorboard import SummaryWriter
import subprocess
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import math
import sys
import utils
import pickle

class FasterModel:
    def __init__(self, base_path="."):

        self.base_path = base_path + "/FasterRCNN"
        
        #subprocess.run(["mkdir", base_path])
        #subprocess.run(["mkdir", base_path + "/logs"])
        #subprocess.run(["mkdir", base_path + "/parameters"])


        self.writer = SummaryWriter(base_path + "/logs")

        #ToDo: creare cartella parameters e anche FasterRCNN 
        self.model_path = self.base_path + "/parameters"

        self.epoch = 0
        self.last_batch = -1 

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        self.model.to(self.device)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(self.params, lr=0.001, momentum=0.9, weight_decay=0.0005)



    
    def train(self, data_loader, print_freq, scaler=None, save_freq=None):
        self.model.train()
        self.metric_logger = utils.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{self.epoch}]"

        lr_scheduler = None
        if self.epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )
        batches_since_last_save = 0
        for batch_idx, (images, targets) in enumerate(self.metric_logger.log_every(data_loader, print_freq, header)):

            if batch_idx <= self.last_batch:
                continue

            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict_reduced)
                sys.exit(1)

            self.optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(losses).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                losses.backward()
                self.optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            self.metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            self.metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

            self.last_batch = batch_idx

            if save_freq is not None:
                batches_since_last_save += 1
                if batches_since_last_save >= save_freq:
                    self.save_model()
                    batches_since_last_save = 0   

        self.save_model()
        self.epoch += self.epoch + 1
        self.last_batch = -1
        return self.metric_logger


    def save_model(self):
        torch.save({
                    'epoch': self.epoch,
                    'last_batch': self.last_batch,
                    'metric_logger': self.metric_logger,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, self.model_path + "/epoch_" + str(self.epoch) + "_batch_" + str(self.last_batch) + ".pth")
        print(f"Model saved at epoch {self.epoch} and batch {self.last_batch}")
        
    def load_model(self, epoch, last_batch):
        self.epoch = epoch
        self.last_batch = last_batch
        diz = torch.load(self.model_path + "/epoch_" + str(self.epoch) + "_batch_" + str(self.last_batch) + ".pth", pickle_module=pickle)
        self.metric_logger = diz['metric_logger']
        self.model.load_state_dict(diz['model_state_dict'])
        self.optimizer.state_dict = diz['optimizer_state_dict']
        print(f"Model loaded at epoch {self.epoch} and batch {self.last_batch}")
