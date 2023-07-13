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
import signal
import threading
import wandb
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from constants import *
#pulire import

class FasterModel:


    #Parameters to model parameters from file or from cloud.
    # load_dict if specified must be a dictionary with the following keys:
    # load_from_wandb: bool (True if you want to load the model from wandb, False if you want to load it from a local path)
    # wandb_entity: string (the entity of the wandb project)
    # wandb_project: string (the name of the wandb project)
    # epoch: int (the epoch from which you want to load the model)
    # batch: int (the batch from which you want to load the model)

    #Parameter to do the logging of the model and its losses to wandb.
    # wandb_logging if specified must be a dictionary with the following keys:
    # wandb_api_key: string (the api key of your wandb account)
    # wandb_entity: string (the entity of the wandb project)
    # wandb_project: string (the name of the wandb project)
    #In this case is not necessary that the project exists, it will be created automatically.
    #Moreover will be created a run for each epoch, so you can see the loss of each epoch separately.

    #data_loader;
    #logging_base_path: string (the base path where the model parameters and the tensorboard logs will be saved)
    def __init__(self, data_loader, logging_base_path=".", wandb_logging=None, load_dict=None):
        
        self.wandb_logging = wandb_logging
        #authenticating to wandb
        if self.wandb_logging is not None:
            wandb.login(key=self.wandb_logging["wandb_api_key"])

        #Defining the directories structure that will contain the logs
        self.data_loader = data_loader
        self.logging_base_path = logging_base_path + "/FasterRCNN_Logging"
        self.tensorboard_logs_path = self.logging_base_path + "/Tensorboard_logs"
        self.tensorboard_logs_all_runs_path = self.tensorboard_logs_path + "/All_Epochs"
        self.model_params_path = self.logging_base_path + "/Model_parameters"

        if not os.path.exists(self.logging_base_path):
            os.makedirs(self.logging_base_path)
            os.makedirs(self.tensorboard_logs_path)
            os.makedirs(self.model_params_path)
            os.makedirs(self.tensorboard_logs_all_runs_path)

        #Two indexes to keep track of the current epoch and batch
        self.epoch = 0
        self.last_batch = 0

        #Defining the device to use
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        #Defining the model with the pretrained weights from Coco
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        self.model.to(self.device)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.SGD(self.params, lr=0.001, momentum=0.9, weight_decay=0.0005)

        self.metric_logger = utils.MetricLogger(delimiter="  ")
        if load_dict is None:
            #Defining a new learning scheduler
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(self.data_loader) - 1)
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters)
            
            self.metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        else: #(The learning scheduler will be resumed if the model is loaded, inside the load functions)
            #load the model from wandb
            if load_dict["load_from_wandb"]:
                self.__load_model_from_wandb(load_dict["epoch"], load_dict["batch"],
                                      load_dict["wandb_entity"], load_dict["wandb_project"])
            else: #load model from file
                self.__load_model(load_dict["epoch"],
                                load_dict["batch"])

        self.current_images = None
        self.current_targets = None

    # Warning: if the training is interrupted and the epoch was not finished,
    #it is necessary to redifine the dataset to skip the element that have been seen, 
    #before calling again train.
    #The code is a variation of the code avaible at this link: https://github.com/pytorch/vision/blob/main/references/detection/engine.py
    def train(self, print_freq, scaler=None, save_freq=None):

        #Creates the directories to log the model
        if not os.path.exists(self.tensorboard_logs_path + "/Epoch_" + str(self.epoch)):
            os.makedirs(self.tensorboard_logs_path +
                        "/Epoch_" + str(self.epoch))

        #Init the writers for tensorboard
        self.writer = SummaryWriter(self.tensorboard_logs_path + "/Epoch_" + str(self.epoch))
        self.writer_all_epoch = SummaryWriter(self.tensorboard_logs_all_runs_path)

        #Init or resume a run if wand_logging
        if self.wandb_logging is not None:
            wandb_api = wandb.Api()

            # Search for the project in the entity
            self.projects = wandb_api.projects(
                self.wandb_logging["wandb_entity"])
            self.project_exists = any(
                p.name == self.wandb_logging["wandb_project"] for p in self.projects)

            # If the project exists...
            if self.project_exists:
                # Search for the run in the project
                self.runs = wandb_api.runs(
                    self.wandb_logging["wandb_entity"] + "/" + self.wandb_logging["wandb_project"])
                run_id = next((run.id for run in self.runs if run.name == (
                    "Epoch_" + str(self.epoch))), None)

                # If the run doesn't exist, create it
                if (run_id is None):
                    print("Wandb: creating new run for epoch " + str(self.epoch))
                    wandb.init(project=self.wandb_logging["wandb_project"],
                               name=("Epoch_" + str(self.epoch)), sync_tensorboard=True)
                # If the run exists, resume it
                else:
                    print("Wandb: resuming run for epoch " + str(self.epoch))
                    wandb.init(project=self.wandb_logging["wandb_project"],
                               id=run_id, resume="must", sync_tensorboard=True)
            # If the project doesn't exist, create it
            else:
                print("Wandb: creating new project and run for epoch " +
                      str(self.epoch))
                wandb.init(project=self.wandb_logging["wandb_project"],
                           name=("Epoch_" + str(self.epoch)),
                           config={
                               "learning_rate": 0.001,
                               "architecture": "FasterRCNN",
                }, sync_tensorboard=True)

        self.model.train()
        header = f"Epoch: [{self.epoch}]"
        batches_since_last_save = 0 #used for controlling the saving frequency

        try:
            # For CTRL+C handling. 
            self.current_images = None
            self.current_targets = None
            for (images, targets) in self.metric_logger.log_every(self.data_loader, print_freq, header, resume_index=self.last_batch):
                
                #read images and targets, and convert them to device
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device)
                            for k, v in t.items()} for t in targets]

                # For CTRL+C handling. 
                self.current_images = images
                self.current_targets = targets

                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                loss_value = losses_reduced.item()

                #Tensorboard log
                self.writer.add_scalar(
                    'loss/train', loss_value, global_step=self.last_batch)
                self.writer_all_epoch.add_scalar(
                    'loss/train', loss_value, global_step=(self.last_batch + len(self.data_loader) * self.epoch))
                #Wandb log
                if self.wandb_logging:
                    wandb.log({"loss": loss_value})

                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")
                    print(loss_dict_reduced)
                    return

                self.optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(losses).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    losses.backward()
                    self.optimizer.step()

                self.lr_scheduler.step()

                self.metric_logger.update(
                    loss=losses_reduced, **loss_dict_reduced)
                self.metric_logger.update(
                    lr=self.optimizer.param_groups[0]["lr"])

                self.last_batch += 1

                if save_freq is not None:
                    batches_since_last_save += 1
                    if batches_since_last_save >= save_freq:
                        self.save_model()
                        batches_since_last_save = 0

                del images
                del targets
                torch.cuda.empty_cache()

                # for CTRL+C handling
                self.current_images = None
                self.current_targets = None
        except KeyboardInterrupt: #Handle the CTRL+C event to clean memory and closing logs
            print("\nCtrl+C pressed. Performing cleanup...")
            del self.current_images
            del self.current_targets
            torch.cuda.empty_cache()

            self.writer.close()
            self.writer_all_epoch.close()

            if self.wandb_logging is not None:
                wandb.finish()

            return

        self.epoch += 1
        self.last_batch = 0
        #Finish the current run and create a new run (for the next epoch)
        if self.wandb_logging is not None:
            wandb.finish()
            wandb_api = wandb.Api()

            # Search for the project in the entity
            self.projects = wandb_api.projects(
                self.wandb_logging["wandb_entity"])
            self.project_exists = any(
                p.name == self.wandb_logging["wandb_project"] for p in self.projects)

            # If the project exists...
            if self.project_exists:
                # Search for the run in the project
                self.runs = wandb_api.runs(
                    self.wandb_logging["wandb_entity"] + "/" + self.wandb_logging["wandb_project"])
                run_id = next((run.id for run in self.runs if run.name == (
                    "Epoch_" + str(self.epoch))), None)

                # If the run doesn't exist, create it
                if (run_id is None):
                    print("Wandb: creating new run for epoch " + str(self.epoch))
                    wandb.init(project=self.wandb_logging["wandb_project"],
                               name=("Epoch_" + str(self.epoch)), sync_tensorboard=True)
                # If the run exists, resume it
                else:
                    print("Wandb: resuming run for epoch " + str(self.epoch))
                    wandb.init(project=self.wandb_logging["wandb_project"],
                               id=run_id, resume="must", sync_tensorboard=True)
            # If the project doesn't exist, create it
            else:
                print("Wandb: creating new project and run for epoch " +
                      str(self.epoch))
                wandb.init(project=self.wandb_logging["wandb_project"],
                           name=("Epoch_" + str(self.epoch)),
                           config={
                               "learning_rate": 0.001,
                               "architecture": "FasterRCNN",
                }, sync_tensorboard=True)

        self.save_model() #Save the model in file or on wandb
        #close all logs
        self.writer.close()
        if self.wandb_logging:
            wandb.finish()
        self.metric_logger = utils.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter(
            "lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        return self.metric_logger

    #Save the current model parameters to file,
    #and if wandb_logging is specified also to wandb.
    def save_model(self):
        #Save in local storage
        torch.save({
            'epoch': self.epoch,
            'last_batch': self.last_batch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'metric_logger': {'meters': self.metric_logger.meters, 'iter_time': self.metric_logger.iter_time, 'data_time': self.metric_logger.data_time},
        }, self.model_params_path + "/epoch_" + str(self.epoch) + "_batch_" + str(self.last_batch) + ".pth")

        #Upload to wandb
        if self.wandb_logging:
            wandb.save(
                self.model_params_path + "/epoch_" +
                str(self.epoch) + "_batch_" + str(self.last_batch) + ".pth",
                base_path=self.model_params_path,
                policy="now"
            )
            print("Model uploaded to wandb.")
            
        print(f"Model saved at epoch {self.epoch} and batch {self.last_batch}")

    #Loads the model called f"epoch_{epoch}_batch_{last_batch}.pth" from the model_params_path
    #It is a PRIVATE method: should not be called outside the class.
    def __load_model(self, epoch, last_batch):
        self.epoch = epoch
        self.last_batch = last_batch
        
        model_path = self.model_params_path + "/epoch_" + str(self.epoch) + "_batch_" + str(self.last_batch) + ".pth"
        
        diz = torch.load(model_path, map_location=self.device)

        self.model.load_state_dict(diz['model_state_dict'])
        self.optimizer.load_state_dict = diz['optimizer_state_dict']

        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                              start_factor=diz["lr_scheduler_state_dict"]["start_factor"],
                                                              total_iters=diz["lr_scheduler_state_dict"]["total_iters"])
        #resume the learning scheduler state
        for i in range(0, diz["lr_scheduler_state_dict"]["last_epoch"]):
            self.lr_scheduler.step()

        #resume the metric logger state
        self.metric_logger.meters = diz['metric_logger']['meters']
        self.metric_logger.iter_time = diz['metric_logger']['iter_time']
        self.metric_logger.data_time = diz['metric_logger']['data_time']

        print(f"Model loaded at epoch {self.epoch} and batch {self.last_batch}")

    #Loads the model called f"epoch_{epoch}_batch_{last_batch}.pth" from wandb, 
    #with entity and project_name as parameters.
    #It is a PRIVATE method: should not be called outside the class.
    def __load_model_from_wandb(self, epoch, last_batch, entity, project_name):
        #Download the model from wandb only if it is not already in the model_params_path
        if not os.path.exists(self.model_params_path + "/epoch_" + str(self.epoch) + "_batch_" + str(self.last_batch) + ".pth"):
            api = wandb.Api()

            #Finds all runs avaible in the project
            runs = api.runs(entity + "/" + project_name)
            #Finds, if exists, the run with the name "Epoch_{epoch}", and obtains its id
            run_id = next((run.id for run in runs if run.name ==("Epoch_" + str(epoch))), None)
            if run_id is None:
                print("No run with this epoch found.")
                return

            #Download the model from wandb
            run = api.run(entity + "/" + project_name + "/" + run_id)
            file = run.file("epoch_" + str(epoch) + "_batch_" + str(last_batch) + ".pth")
            if file.size != 0:
                file.download(self.model_params_path, replace=True)
            else:
                return

        #Once the model is downloaded, use the __load_model method to load it
        self.__load_model(epoch, last_batch)

    #Combine all the runs of the active project in a single run called "All Epochs".
    #This run will contain all the losses of the single runs, and can be used to plot the losses in a single curve.
    def combine_all_epochs(self):
        if not self.wandb_logging:
            return
        
        wandb_api = wandb.Api()
        cumulative_run = "All Epochs"

        # Search for the project in the entity
        projects = wandb_api.projects(self.wandb_logging["wandb_entity"])
        project_exists = any(p.name == self.wandb_logging["wandb_project"] for p in projects)
        
        # If the project doesn't exists...
        if not project_exists:
            return
        
        # Search for the run in the project
        wandb_path = self.wandb_logging["wandb_entity"] + "/" + self.wandb_logging["wandb_project"]
        runs = wandb_api.runs(wandb_path)
        run_id = next((run.id for run in runs if run.name == (cumulative_run)), None)
        
        # If the run exist delete
        if (run_id is not None):
            print(f"Wandb: Deleting run {cumulative_run}")
            wandb_api.run(wandb_path + "/" + run_id).delete()
        
        # Create the run
        print(f"Creating run {cumulative_run}")
        wandb.init(project=self.wandb_logging["wandb_project"], name=cumulative_run)

        losses = []
        for i in range(0, len(runs)): #Put all losses in a single list
            run_id = next((run.id for run in runs if run.name == (
                                "Epoch_" + str(i))), None)
            if (run_id is None):
                break
            
            run = wandb_api.run(wandb_path + "/" + run_id)
            lista = run.history(pandas=False)
            for d in lista:
                losses.append(d["loss"])

        for l in losses: #Log all losses in the run
            wandb.log({"loss": l})
        wandb.finish()
    
    #If catIds is None, evaluates the mean metrics on all the classes;
    #If catIds is "all", evaluates the single metrics for each of the classes;
    #If catIds is a list of integers, evaluates the single metrics for each of the classes 
    #   with the specified ids.

    #The code is a variation of the code avaible at this link: https://github.com/pytorch/vision/blob/main/references/detection/engine.py
    @torch.inference_mode()
    def evaluate(self, data_loader, catIds=None):
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        self.model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"

        # Convert the dataset into a COCO dataset format
        coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = self._get_iou_types() #type of metrics supported by current model

        
        coco_evaluator_list = []
        #The first coco_evaluator will contain the mean metrics of all classes
        coco_evaluator_list.append(CocoEvaluator(coco, iou_types))

        #Here are created as many coco_evaluator as the number of classes
        if catIds is not None:
            if catIds == "all":
                catIds = list(range(0, len(str_label)))
            for i in range(0, len(catIds)):
                coco_evaluator_list.append(CocoEvaluator(coco, iou_types))
                coco_evaluator_list[i + 1].coco_eval['bbox'].params.catIds = [catIds[i]]


        for images, targets in metric_logger.log_every(data_loader, 5, header):
            images = list(img.to(self.device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            outputs = self.model(images)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()}
                       for t in outputs]
            model_time = time.time() - model_time

            # Dictionary containg the predictions for each image.
            # The key is the image_id and the value is the prediction
            res = {target["image_id"].item(): output for target,
                   output in zip(targets, outputs)}

            evaluator_time = time.time()
            for coco_evaluator in coco_evaluator_list:
                coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time,
                                 evaluator_time=evaluator_time)
            
            #Free Memory, important if using Gpu
            del images
            del targets
            torch.cuda.empty_cache()

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        for coco_evaluator in coco_evaluator_list:
            coco_evaluator.synchronize_between_processes()
        
        
        # accumulate predictions from all images
        for i, coco_evaluator in enumerate(coco_evaluator_list):
            if i != 0:
                print(f"Metrics relative to the class {str_label[catIds[i - 1]]}")
            else:
                print("Mean metrics for all classes")
            coco_evaluator.accumulate()
            coco_evaluator.summarize()
            print("\n\n")

        torch.set_num_threads(n_threads)
        return coco_evaluator

    # Return a list containing all the types of IOU that are supported
    # by the model. (In our case it will return only bbox)
    def _get_iou_types(self):
        model_without_ddp = self.model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = self.model.module
        iou_types = ["bbox"]
        if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
            iou_types.append("segm")
        if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
            iou_types.append("keypoints")
        print("IOU types: ", iou_types)
        return iou_types

    #Shows the given image with it's predicted bounding boxes.
    #image: a tensor on wich to apply the current model;
    #treshold: the treshold to filter predictions that have a low score
    #classes_to_show: list of integers. They are the classes that will be shown 
    #return_fig: if True will return the fig object
    #on the final image. In None all classes will be shown.
    @torch.inference_mode()
    def apply_object_detection(self, image, treshold=0.5, classes_to_show=None, return_fig=False):
        self.model.eval()
        image = image.unsqueeze(0)
        
        #apply the model, and obtain the predictions
        with torch.no_grad():
            predictions = self.model(image.clone().to(self.device))
        predictions = [{k: v.to(torch.device("cpu")) for k, v in t.items()}
                       for t in predictions]
        
        boxes = predictions[0]['boxes']
        labels = predictions[0]['labels']
        scores = predictions[0]['scores']

        fig, ax = plt.subplots(1)
        ax.imshow(image[0].permute(1, 2, 0))

        #Add each prediction to the final image
        for i, (l, bbox) in enumerate(zip(labels, boxes)):
            if scores[i] < treshold:
                continue
            if (classes_to_show is not None and l not in classes_to_show):
                continue
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            x = bbox[0]
            y = bbox[1]

            # Create a rectangle patch
            if (l < len(str_label)):
                rect = patches.Rectangle(
                    (x, y), width, height, linewidth=1, edgecolor=colors_bounding[l], facecolor="none")
                ax.text(x, y - 10, str_label[l], color=colors_bounding[l])
            else:
                rect = patches.Rectangle(
                    (x, y), width, height, linewidth=1, edgecolor="black", facecolor="none")
                ax.text(x, y - 10, "Unknown", color="black")
            # Add the rectangle to the axes
            ax.add_patch(rect)

        if return_fig:
            return fig
        else:
            # Show the plot
            plt.show()

