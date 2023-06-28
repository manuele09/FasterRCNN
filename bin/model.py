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


class FasterModel:

    def handle_interrupt(self, signal, frame):
        print("\nCtrl+C pressed. Performing cleanup...")
        del self.current_images
        del self.current_targets
        torch.cuda.empty_cache()
        thread_id = threading.current_thread().ident
        print("Thread ID:", thread_id)

        return
    # La load sarà possibile farla SOLO chiamando la __init__.
    # Se non faccio la load posso istanziare lo scheduler, altrimenti lo istanzio nella load.

    # load_dict if specified must be a dictionary with the following keys:
    # load_from_wandb: bool (True if you want to load the model from wandb, False if you want to load it from a local path)
    # wandb_entity: string (the entity of the wandb project)
    # wandb_project: string (the name of the wandb project)
    # epoch: int (the epoch from which you want to load the model)
    # batch: int (the batch from which you want to load the model)
    # path: string (the path from which you want to load the model, if load_from_wandb is False;
    #               if None the path will be the default one, ie logging_base_path ...)

    # wandb_logging if specified must be a dictionary with the following keys:
    # wandb_api_key: string (the api key of your wandb account)
    # wandb_entity: string (the entity of the wandb project)
    # wandb_project: string (the name of the wandb project)

    def __init__(self, data_loader, logging_base_path=".", wandb_logging=None, load_dict=None, save_memory=False):
        self.wandb_logging = wandb_logging
        self.save_memory = save_memory

        if self.wandb_logging is not None:  
            wandb.login(key=self.wandb_logging["wandb_api_key"])
            
        self.data_loader = data_loader
        self.logging_base_path = logging_base_path + "/FasterRCNN_Logging"
        self.tensorboard_logs_path = self.logging_base_path + "/Tensorboard_logs"
        self.model_params_path = self.logging_base_path + "/Model_parameters"

        if not os.path.exists(self.logging_base_path):
            os.makedirs(self.logging_base_path)
            os.makedirs(self.tensorboard_logs_path)
            os.makedirs(self.model_params_path)
            # mettere la stringa in una self.
            os.makedirs(self.tensorboard_logs_path + "/All_Epochs")

        self.epoch = 0
        self.last_batch = 0

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights='DEFAULT')
        self.model.to(self.device)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            self.params, lr=0.001, momentum=0.9, weight_decay=0.0005)

        self.metric_logger = utils.MetricLogger(delimiter="  ")
        if load_dict is None:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(self.data_loader) - 1)
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )
            self.metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        else:
            if load_dict["load_from_wandb"]:
                self.load_model_wandb(load_dict["epoch"], load_dict["batch"],
                                      load_dict["wandb_entity"], load_dict["wandb_project"], load_dict["path"])
            else:
                self.load_model(load_dict["epoch"],
                                load_dict["batch"], load_dict["path"])


        # signal.signal(signal.SIGINT, self.handle_interrupt)
        self.current_images = None
        self.current_targets = None



    #ATTENZIONE: Una volta che il training è stato interrotto e lo si vuole riprendere
    #bisogna creare un NUOVO Dataset che escluda le immagini già processate.
    def train(self, print_freq, scaler=None, save_freq=None):

        if not os.path.exists(self.tensorboard_logs_path + "/Epoch_" + str(self.epoch)):
            os.makedirs(self.tensorboard_logs_path +
                        "/Epoch_" + str(self.epoch))

        self.writer = SummaryWriter(
            self.tensorboard_logs_path + "/Epoch_" + str(self.epoch))
        self.writer_all_epoch = SummaryWriter(
            self.tensorboard_logs_path + "/All_Epochs")

        if self.wandb_logging is not None:
            wandb_api = wandb.Api()

            # Search for the project in the entity
            self.projects = wandb_api.projects(self.wandb_logging["wandb_entity"])
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
                print("Wandb: creating new project and run for epoch " + str(self.epoch))
                wandb.init(project=self.wandb_logging["wandb_project"],
                           name=("Epoch_" + str(self.epoch)),
                           config={
                               "learning_rate": 0.001,
                               "architecture": "FasterRCNN",
                           }, sync_tensorboard=True)

        self.model.train()
        header = f"Epoch: [{self.epoch}]"

        batches_since_last_save = 0

        try:
            for (images, targets) in self.metric_logger.log_every(self.data_loader, print_freq, header, resume_index=self.last_batch):

                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device)
                            for k, v in t.items()} for t in targets]

                # For CTRL+C handling
                self.current_images = images
                self.current_targets = targets

                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(
                    loss for loss in loss_dict_reduced.values())

                loss_value = losses_reduced.item()

                self.writer.add_scalar(
                    'loss/train', loss_value, global_step=self.last_batch)
                self.writer_all_epoch.add_scalar(
                    'loss/train', loss_value, global_step=(self.last_batch + len(self.data_loader) * self.epoch))
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
        except KeyboardInterrupt:
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
        self.save_model()
        self.writer.close()
        if self.wandb_logging:
            wandb.finish()
        self.metric_logger = utils.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter(
            "lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        return self.metric_logger

    def save_model(self):
        torch.save({
            'epoch': self.epoch,
            'last_batch': self.last_batch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'metric_logger': {'meters': self.metric_logger.meters, 'iter_time': self.metric_logger.iter_time, 'data_time': self.metric_logger.data_time},
        }, self.model_params_path + "/epoch_" + str(self.epoch) + "_batch_" + str(self.last_batch) + ".pth")

        if self.wandb_logging:
            wandb.save(
                self.model_params_path + "/epoch_" +
                str(self.epoch) + "_batch_" + str(self.last_batch) + ".pth",
                base_path=self.model_params_path,
                policy="now"
            )
            print("Model uploaded to wandb.")
            if self.save_memory:
                os.remove(self.model_params_path + "/epoch_" +
                          str(self.epoch) + "_batch_" + str(self.last_batch) + ".pth")
            # aggiungere opzione per eliminare il file in locale
        print(f"Model saved at epoch {self.epoch} and batch {self.last_batch}")

    def load_model(self, epoch, last_batch, path=None):
        self.epoch = epoch
        self.last_batch = last_batch

        if path is None:
            path = self.model_params_path

        diz = torch.load(path + "/epoch_" +
                         str(self.epoch) + "_batch_" + str(self.last_batch) + ".pth")

        self.model.load_state_dict(diz['model_state_dict'])
        self.optimizer.load_state_dict = diz['optimizer_state_dict']

        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                              start_factor=diz["lr_scheduler_state_dict"]["start_factor"],
                                                              total_iters=diz["lr_scheduler_state_dict"]["total_iters"])
        for i in range(0, diz["lr_scheduler_state_dict"]["last_epoch"]):
            self.lr_scheduler.step()

        self.metric_logger.meters = diz['metric_logger']['meters']
        self.metric_logger.iter_time = diz['metric_logger']['iter_time']
        self.metric_logger.data_time = diz['metric_logger']['data_time']

        print(
            f"Model loaded at epoch {self.epoch} and batch {self.last_batch}")

    def load_model_wandb(self, epoch, last_batch, entity, project_name, path=None):

        api = wandb.Api()

        runs = api.runs(entity + "/" + project_name)
        run_id = next((run.id for run in runs if run.name ==
                      ("Epoch_" + str(epoch))), None)
        if run_id is None:
            print("No run with this epoch found.")
            return

        run = api.run(entity + "/" + project_name + "/" + run_id)
        run.file("epoch_" + str(epoch) + "_batch_" + str(last_batch) +
                 ".pth").download(self.model_params_path, replace=True)

        self.load_model(epoch, last_batch, path)

        if self.save_memory:
            os.remove(self.model_params_path + "/epoch_" +
                      str(self.epoch) + "_batch_" + str(self.last_batch) + ".pth")

    @torch.inference_mode()
    def evaluate(self, data_loader):
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        self.model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"

        # Convert the dataset into a COCO dataset format
        coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = self._get_iou_types()
        coco_evaluator = CocoEvaluator(coco, iou_types)

        for images, targets in metric_logger.log_every(data_loader, 100, header):
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
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time,
                                 evaluator_time=evaluator_time)

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

        print("Max detects")
        print(
            coco_evaluator.coco_eval['bbox'].eval['precision'][0, 0, 0, 0, 1])

        # [10, 101, 5, 4, 3]
        # [T x R  x K xA xM]
        # A = 4 tipi di aree
        # M = 3 treshold max detections

        # K = 5 tipi di categorie
        # R = 101 recall thresholds for evaluation
        # T = 10 IoU thresholds for evaluation

        # idea: consideriamo coco_evaluator.coco_eval['bbox'].eval['precision']
        # E' un numpy array di 5 dimensioni (vedi sopra)
        # La terza dimensione rappresenta la classe
        # Possiamo rendere -1 tutti i suoi elementi che non corrispondono alla classe
        # che ci interessa, di questo modo possiamo chiamare co_evaluator.coco_eval['bbox'].summarize()
        # che farà tutto come al solito, ma filtrerà gli elementi che non sono maggiori di -1

        print("Prima Categoria:")
        coco_evaluator.coco_eval['bbox'].eval['precision'] = coco_evaluator.coco_eval['bbox'].eval['precision'][:, :, 0:1, :, :]
        coco_evaluator.coco_eval['bbox'].eval['precision'] = np.ones(
            coco_evaluator.coco_eval['bbox'].eval['precision'].shape)

        coco_evaluator.coco_eval['bbox'].summarize()
        print(coco_evaluator.coco_eval['bbox'].eval['precision'].shape)

        torch.set_num_threads(n_threads)
        return coco_evaluator

    # Return a list containing all the types of IOU that are supported
    # by the model. (In our case only bbox)

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
