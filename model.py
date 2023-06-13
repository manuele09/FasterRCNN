import torch
import torchvision
import datetime
from custom_utils import *

class FasterModel:
    def __init__(self, model_path, last_epoch=None, last_batch=None):

        self.model_path = model_path + "/model"

        #last training epoch and batch
        if (last_epoch==None):
            self.last_epoch = -1
        else:
            self.last_epoch = last_epoch
        
        if (last_batch==None):
            self.last_batch = -1
        else:
            self.last_batch = last_batch
        
            

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        self.model.to(self.device)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(self.params, lr=0.001, momentum=0.9, weight_decay=0.0005)


        #print(self.last_epoch)
        if (self.last_epoch!=-1 | self.last_batch):
            diz = torch.load(self.model_path + "_epoch_" + str(self.last_epoch) + "_batch_" + str(self.last_batch) + ".pth")
            self.model.state_dict = diz['model_state_dict']
            #self.optimizer.state_dict = diz['optimizer_state_dict']
            print("loaded")


    

    #train for one epoch
    def train(self, data_loader, train_loss_hist):
        print('Starting training: ' + str(datetime.datetime.now()))
        
        start_time =datetime.datetime.now()

        for i, data in enumerate(data_loader):
            if (i <= self.last_batch):
                continue

            self.optimizer.zero_grad()
            images, targets = data

            images_filtered = []
            targets_filtered = []

            for j in range(len(targets)):
                if targets[j]["boxes"].shape[0] > 0:
                    images_filtered.append(torch.tensor(images[j]).to(self.device))
                    targets_filtered.append({k: v.to(self.device) for k, v in targets[j].items()})
                else:
                    print("Element " + str(i*data_loader.batch_size + j) + " removed.")

            images = torch.stack(images_filtered)
            targets = targets_filtered

            
            print(datetime.datetime.now())
            print("Starting batch " + str(i))
            
            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            train_loss_hist.send(loss_value)
            print("Loss: " + str(loss_value))

            losses.backward()
            self.optimizer.step()
            
            self.last_batch += 1
            self.save_model(self.model_path + "_epoch_" + str(self.last_epoch + 1) + "_batch_" + str(self.last_batch) + ".pth")
            print("Finished batch " + str(i))
            print("\n")
        end_time = datetime.datetime.now()
        print("Total time: " + str(time_diff(start_time, end_time)) + " secondi")
        
        self.last_epoch += 1
        self.last_batch = -1


    def save_model(self, path):
        torch.save({
                    'epoch': self.last_epoch,
                    'batch': self.last_batch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, path)