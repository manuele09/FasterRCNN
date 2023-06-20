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
import time
import os

class FasterModel:
    def __init__(self, logging_base_path="."):

        self.logging_base_path = logging_base_path + "/FasterRCNN_Logging"
        self.tensorboard_logs_path = self.logging_base_path + "/Tensorboard_logs"
        self.model_params_path = self.logging_base_path + "/Model_parameters"

        if not os.path.exists(self.logging_base_path):
            os.makedirs(self.logging_base_path)
            os.makedirs(self.tensorboard_logs_path)
            os.makedirs(self.model_params_path)

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

        self.metric_logger = utils.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))



    
    def train(self, data_loader, print_freq, scaler=None, save_freq=None):
        self.writer = SummaryWriter(self.tensorboard_logs_path, flush_secs=10, purge_step=self.last_batch + 1)


        self.model.train()
        header = f"Epoch: [{self.epoch}]"

        lr_scheduler = None
        if self.epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )
        batches_since_last_save = 0
        for batch_idx, (images, targets) in enumerate(self.metric_logger.log_every(data_loader, print_freq, header, resume_index=self.last_batch)):

            if batch_idx <= self.last_batch:
                if lr_scheduler is not None:
                    lr_scheduler.step()
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

            self.writer.add_scalar('loss/train', loss_value, global_step=batch_idx)
            self.writer.flush()

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
            
            del images
            del targets
            torch.cuda.empty_cache()


        self.epoch += 1
        self.last_batch = -1
        self.save_model()
        self.metric_logger = utils.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        return self.metric_logger


    def save_model(self):
        torch.save({
                    'epoch': self.epoch,
                    'last_batch': self.last_batch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metric_logger': {'meters': self.metric_logger.meters, 'iter_time': self.metric_logger.iter_time, 'data_time': self.metric_logger.data_time},
                    }, self.model_params_path + "/epoch_" + str(self.epoch) + "_batch_" + str(self.last_batch) + ".pth")
       
        # with open(self.model_path + "/metric_logger_epoch_" + str(self.epoch) + "_batch_" + str(self.last_batch) + ".pickle", "wb") as outfile:
        #     pickle.dump(self.metric_logger.meters, outfile)
         

        print(f"Model saved at epoch {self.epoch} and batch {self.last_batch}")
        
    def load_model(self, epoch, last_batch):
        self.epoch = epoch
        self.last_batch = last_batch
        diz = torch.load(self.model_params_path + "/epoch_" + str(self.epoch) + "_batch_" + str(self.last_batch) + ".pth")
        self.model.load_state_dict(diz['model_state_dict'])
        self.optimizer.load_state_dict = diz['optimizer_state_dict']

        self.metric_logger.meters = diz['metric_logger']['meters']
        self.metric_logger.iter_time = diz['metric_logger']['iter_time']
        self.metric_logger.data_time = diz['metric_logger']['data_time']

        # with open(self.model_path + "/metric_logger_epoch_" + str(self.epoch) + "_batch_" + str(self.last_batch) + ".pickle", "rb") as infile:
        #     self.metric_logger.meters = pickle.load(infile)
        print(f"Model loaded at epoch {self.epoch} and batch {self.last_batch}")
    
    @torch.inference_mode()
    def evaluate(self, data_loader):
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        self.model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"

        coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = self._get_iou_types()
        coco_evaluator = CocoEvaluator(coco, iou_types)

        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(img.to(self.device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            outputs = self.model(images)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

            del images
            del targets
            torch.cuda.empty_cache()

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
        return coco_evaluator

    def _get_iou_types(self):
        model_without_ddp = self.model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = self.model.module
        iou_types = ["bbox"]
        if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
            iou_types.append("segm")
        if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
            iou_types.append("keypoints")
        return iou_types

    