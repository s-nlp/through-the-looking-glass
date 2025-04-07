import gc
import os
import time
import torch
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from torch.utils.data import Dataset
from torch.optim import lr_scheduler


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "False"


CONFIG = {"seed": 2023,
          "epochs": 10,
          "train_batch_size": 1,
          "valid_batch_size": 1,
          "learning_rate": 1e-4,
          "scheduler": "CosineAnnealingLR",
          "min_lr": 1e-6,
          "T_max": 500,
          "weight_decay": 1e-6,
          "n_accumulate": 20,
          "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          "competition": "SD",
          }


def set_seed(seed=42):
    """
    REPRODUCIBILITY."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


class ImageCaptioningDataset(Dataset):
    def __init__(self, images, labels, processor):
        self.images = images
        self.labels = labels
        self.processor = processor
        self.prefix = "This image is "

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        text = self.prefix + "weird" if self.labels[idx] else "normal"
        encoding = self.processor(images=self.images[idx], text=text, 
                                  padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        input_ids = data["input_ids"].to(device)
        pixel_values = data["pixel_values"].to(device)

        batch_size = input_ids.size(0)

        outputs = model(input_ids=input_ids, 
                        pixel_values=pixel_values, 
                        labels=input_ids)

        loss = outputs.loss
        loss = loss / CONFIG["n_accumulate"]
        loss.backward()

        if (step + 1) % CONFIG["n_accumulate"] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]["lr"])
    gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch, lr):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for _, data in bar:        
        input_ids = data["input_ids"].to(device)
        pixel_values = data["pixel_values"].to(device)

        batch_size = input_ids.size(0)

        outputs = model(input_ids=input_ids, 
                        pixel_values=pixel_values, 
                        labels=input_ids)

        loss = outputs.loss

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                        LR=lr)
    
    gc.collect()
    
    return epoch_loss


def run_training(model, optimizer, scheduler, train_loader, valid_loader, num_epochs):

    if torch.cuda.is_available():
        print("using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_epoch_loss = np.inf
    history = defaultdict(list)
    summary = {}
    
    for epoch in range(1, num_epochs + 1): 
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CONFIG["device"], epoch=epoch)
        
        val_epoch_loss = valid_one_epoch(model, valid_loader, device=CONFIG["device"], 
                                         epoch=epoch, lr=optimizer.param_groups[0]["lr"])
    
        history["Train Loss"].append(train_epoch_loss)
        history["Valid Loss"].append(val_epoch_loss)
        
        # Log the metrics
        print({"Train Loss": train_epoch_loss})
        print({"Valid Loss": val_epoch_loss})
        
        # deep copy the model
        if val_epoch_loss <= best_epoch_loss:
            print(f"new best validation loss ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            summary["Best Loss"] = best_epoch_loss

        print()
    
    end = time.time()
    time_elapsed = end - start
    print("Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_epoch_loss))

    return model, history, summary


def fetch_scheduler(optimizer):
    if CONFIG["scheduler"] == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG["T_max"], 
                                                   eta_min=CONFIG["min_lr"])
    elif CONFIG["scheduler"] == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG["T_0"], 
                                                             eta_min=CONFIG["min_lr"])
    elif CONFIG["scheduler"] == None:
        return None
        
    return scheduler


@torch.no_grad()
def inference(model, images, device):
    model.eval()
    
    processor = CONFIG["processor"]
    results = []

    for img in images:
        data = processor(images=img, text="This image is ", return_tensors="pt").to(device, torch.float16)

        generated_ids = model.generate(**data)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        if "weird" in generated_text.lower():
            results.append(1)
        elif "normal" in generated_text.lower():
            results.append(0)
        else:
            results.append(-1)

    return results
