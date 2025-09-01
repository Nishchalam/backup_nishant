#from utils import *
import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import re
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import torch.nn.functional as F



device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
# device="cpu"
print(f"Using {device} device")


class ModelCheckpointStore:
    def __init__(self, dump_dir):
        self.dump_dir = dump_dir
        self.best_param_epoch = None
        self.last_save_step = 0

    def __call__(self, model, training_metrics, metric, current_step):
        param = "acc" if metric.endswith("acc") else "loss"
        model_save_dir = os.path.join(self.dump_dir, "checkpoints")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        new_valid = os.path.join(model_save_dir, f"step_{current_step}_{param}_latest.pth")
        old_valid = os.path.join(model_save_dir, f"step_{self.last_save_step}_{param}_latest.pth")
        torch.save(model, new_valid)
        self.last_save_step = current_step

        if os.path.exists(old_valid):
            os.remove(old_valid)

        if len(training_metrics[metric]) == 1:
            best_checkpoint = os.path.join(model_save_dir, f"step_{current_step}_{param}_best.pth")
            torch.save(model.state_dict(), best_checkpoint)
            self.best_param_epoch = current_step
        else:
            is_best = False
            if param == "acc" and training_metrics[metric][-1] > training_metrics[metric][-2]:
                is_best = True
            elif param == "loss" and training_metrics[metric][-1] < training_metrics[metric][-2]:
                is_best = True

            if is_best:
                best_checkpoint = os.path.join(model_save_dir, f"step_{current_step}_{param}_best.pth")
                torch.save(model, best_checkpoint)
                old_best = os.path.join(model_save_dir, f"step_{self.best_param_epoch}_{param}_best.pth")
                if os.path.exists(old_best):
                    os.remove(old_best)
                self.best_param_epoch = current_step




def display(step, training_metrics):
    to_print = f"epoch : {step}\n" + "\n".join([f"{key} : {value[-1]}" for key, value in training_metrics.items()])        
    print(to_print)


def plot(dump_dir, model_name, **kwargs):
    plot_save_path = os.path.join(dump_dir, f"{model_name}_plots")
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    acc_plot = plt.figure(figsize=(15, 8))
    loss_plot = plt.figure(figsize=(15, 8))

    for key, value in kwargs.items():
        if key.endswith('acc'):
            plt.figure(acc_plot.number)
            plt.plot(value, label=key)
            plt.xlabel('steps')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Accuracy')

        elif key.endswith('loss'):
            plt.figure(loss_plot.number)
            plt.plot(value, label=key)
            plt.xlabel('steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss')

    plt.figure(acc_plot.number)
    plt.savefig(os.path.join(plot_save_path, 'acc.png'))
    plt.close(acc_plot)

    plt.figure(loss_plot.number)
    plt.savefig(os.path.join(plot_save_path, 'loss.png'))
    plt.close(loss_plot)


def hz_to_bin(f):
    mask = np.where(f==0.0)
    
    cent = 1200 * np.log2((f / 10) + 1e-9)
    cent -= (2500 - 20)
    cent[mask] = 0.0
    bin_ = np.floor(cent / 20)

    return np.minimum(bin_, 300)


class ConvBlock(nn.Module):
    def __init__(self, f, w, s, in_channels):
        super().__init__()
        p1 = (w - 1) // 2
        p2 = (w - 1) - p1
        self.pad= nn.ZeroPad2d((0, 0, p1, p2))
        
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=f, kernel_size=(w, 1), stride=s)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(f)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x

class CREPE(nn.Module):
    def __init__(self, model_capacity="tiny"):
        super().__init__()

        capacity_multiplier = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
        }[model_capacity]

        self.layers = [1, 2, 3, 4, 5, 6]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        filters = [1] + filters
        widths = [512, 64, 64, 64, 64, 64]
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        for i in range(len(self.layers)):
            f, w, s, in_channel = filters[i+1], widths[i], strides[i], filters[i]
            self.add_module("conv%d" % i, ConvBlock(f, w, s, in_channel))

        self.linear = nn.Linear(64*capacity_multiplier, 300)
        # self.load_weight(model_capacity)
        # self.eval()
        
    # def load_weight(self, model_capacity):
    #     download_weights(model_capacity)
    #     package_dir = os.path.dirname(os.path.realpath(__file__))
    #     filename = "crepe-{}.pth".format(model_capacity)
    #     self.load_state_dict(torch.load(os.path.join(package_dir, filename)))

    def forward(self, x):
        # x : shape (batch, sample)
        x = x.view(x.shape[0], 1, -1, 1)
        # print(x.shape)
        for i in range(len(self.layers)):
            x = self.__getattr__("conv%d" % i)(x)
            # print(x.shape)

        x = x.permute(0, 3, 2, 1)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x
# def collate_fn(batch):
#     src_batch, tgt_batch = [], []
#     for src_sample, tgt_sample in batch:
#         src_batch.append(torch.tensor(np.load(src_sample), dtype=torch.float32))
#         tgt_batch.append(torch.tensor(hz_to_bin(np.load(tgt_sample)), dtype=torch.long))

#     # Instead of concatenating, pad the sequences
#     src_batch = pad_sequence(src_batch, batch_first=True).to(device)
#     tgt_batch = pad_sequence(tgt_batch, batch_first=True).to(device)
    
#     return src_batch, tgt_batch

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_frames = np.load(src_sample)  # Load the entire file
        tgt_frames = hz_to_bin(np.load(tgt_sample))  # Convert target frequencies to bins
        
        for i in range(src_frames.shape[0]):  # Iterate over each frame
            src_batch.append(torch.tensor(src_frames[i], dtype=torch.float32))
            tgt_batch.append(torch.tensor(tgt_frames[i].astype(np.int64), dtype=torch.long))

    
    src_batch = torch.stack(src_batch).to(device)  # Stack frames into a batch
    tgt_batch = torch.stack(tgt_batch).to(device)
    
    return src_batch, tgt_batch


data_path = "/speech/nishanth/raw_exps/full_final_data_crepe_16khz"
train_X = os.listdir(f"{data_path}/train/raw")
train_X = [os.path.join(f"{data_path}/train/raw", x) for x in train_X]
train_y = os.listdir(f"{data_path}/train/labels")
train_y = [os.path.join(f"{data_path}/train/labels", y) for y in train_y]
valid_X = os.listdir(f"{data_path}/valid/raw")
valid_X = [os.path.join(f"{data_path}/valid/raw", x) for x in valid_X]
valid_y = os.listdir(f"{data_path}/valid/labels")
valid_y = [os.path.join(f"{data_path}/valid/labels", y) for y in valid_y]

train_data = list(zip(train_X, train_y))
valid_data = list(zip(valid_X, valid_y))

batch_size = 16  # Number of frames per batch
train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, collate_fn=collate_fn)


def train_and_eval(model, train_loader, valid_loader, optimizer, criterion, max_steps, device, dump_dir, model_name, metric, last_epoch=0):
    
    checkpint_store = ModelCheckpointStore(dump_dir)
    
    training_metrics = {"train_avg_loss":[],
                        "train_avg_acc":[],
                        "valid_avg_loss":[],
                        "valid_avg_acc":[]}
    for step in range(last_epoch, max_steps+last_epoch):
        train_step_loss = 0
        train_step_correct_pred = 0
        valid_step_loss = 0
        valid_step_correct_pred = 0
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.squeeze(0).to(device), y.squeeze(0).to(device)
            # print("aaa", x.shape, y.shape)
            optimizer.zero_grad()
            train_out = model(x)
            # print("bbb", train_out.shape)
            loss = criterion(train_out.reshape(-1, 300), y.flatten())
            loss.backward()
            optimizer.step()
            train_step_loss += loss.item()
            _, train_correct_outputs = torch.max(train_out.reshape(-1, 300), dim=1)
            train_step_correct_pred += (train_correct_outputs == y.flatten()).sum().item()        
        training_metrics["train_avg_loss"].append(train_step_loss / len(train_loader))
        training_metrics["train_avg_acc"].append(train_step_correct_pred / len(train_loader))
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (x_val, y_val) in enumerate(valid_loader):
                # x_val = torch.transpose(x_val, 0, 1)
                # y_val = torch.transpose(y_val, 0, 1)
                x_val, y_val = x_val.squeeze(0).to(device), y_val.squeeze(0).to(device)
                valid_out = model(x_val)
                valid_loss = criterion(valid_out.reshape(-1, 300), y_val.flatten())
                valid_step_loss += valid_loss.item()
                _, valid_correct_outputs = torch.max(valid_out.reshape(-1, 300), dim=1)
                valid_step_correct_pred += (valid_correct_outputs == y_val.flatten()).sum().item()

            training_metrics["valid_avg_loss"].append(valid_step_loss / len(valid_loader))
            training_metrics["valid_avg_acc"].append(valid_step_correct_pred / len(valid_loader))
                 
        display(step, training_metrics)
        checkpint_store(model, training_metrics, metric, step)
        plot(dump_dir, model_name, **training_metrics)




model = CREPE().to(device)

cross_entropy_loss_weights = torch.full((300,), 2.5)
cross_entropy_loss_weights[0] = 1
cross_entropy_loss_weights = cross_entropy_loss_weights.float().to(device)
loss_fn = nn.CrossEntropyLoss(weight=cross_entropy_loss_weights)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
epochs = 100

dump_dir = "/speech/nishanth/raw_exps/dump_crepe"
os.makedirs(dump_dir, exist_ok=True)
last_model_save_epoch = 0

model_name = "weighted_CEL_one_radius_crepe"
train_and_eval(model, train_dataloader, valid_dataloader, optimizer, loss_fn, epochs, device, dump_dir, model_name, "valid_avg_loss", last_epoch=0)
