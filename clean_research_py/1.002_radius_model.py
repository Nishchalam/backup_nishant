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
# device = "cpu"
print(f"Using {device} device")

data_path = "/speech/nishanth/clean_research/final_data_1.01_1.03_1.06_ptdb"

train_X = os.listdir(f"{data_path}/train_eq/gd_all")
train_X = [os.path.join(f"{data_path}/train_eq/gd_all", x) for x in train_X]
train_y = os.listdir(f"{data_path}/train_eq/labels")
train_y = [os.path.join(f"{data_path}/train_eq/labels", y) for y in train_y]

valid_X = os.listdir(f"{data_path}/valid_eq/gd_all")
valid_X = [os.path.join(f"{data_path}/valid_eq/gd_all", x) for x in valid_X]
valid_y = os.listdir(f"{data_path}/valid_eq/labels")
valid_y = [os.path.join(f"{data_path}/valid_eq/labels", y) for y in valid_y]

#List of tuples [ (input_link, label_link)]
# e.g [ ('fake_gd_data/train/group_delay/1.npy', 'fake_gd_data/train/labels/1.npy') ]
train_data = list(zip(train_X, train_y))
valid_data = list(zip(valid_X, valid_y))

"""    - - - - - - - - - - - - - - - - - - - - - -    """

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

        # if os.path.exists(old_valid):
        #     os.remove(old_valid)

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
    cent -= (1997.37 - 20)
    cent[mask] = 0.0
    bin_ = np.floor(cent / 20)

    return np.minimum(bin_, 300)


PAD_IDX=0
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(torch.tensor(np.load(src_sample), dtype=torch.float32)[0,:,:])
        tgt_batch.append(torch.tensor(hz_to_bin(np.load(tgt_sample)), dtype=torch.long))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=False)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=False)
    return src_batch.to(device), tgt_batch.to(device)

batch_size = 32
train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, collate_fn=collate_fn)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(d_model)
        
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.linear1 = nn.Linear(d_model, 512)
        self.linear2 = nn.Linear(512, 300)
        self.dropout_layer = nn.Dropout(dropout)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(512)

    def forward(self, x):
        # Apply convolutional layers with batch normalization
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, 512, n_frames)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, n_frames, d_model)
        
        # Apply positional encoding and Transformer encoder
        x = self.pos_encoder(x)
        out = self.encoder(x)

        # Apply linear layers and normalization
        out = self.layer_norm1(out)
        out = F.relu(self.linear1(out))
        out = self.layer_norm2(out)
        out = self.dropout_layer(out)
        logits = self.linear2(out)
        return logits


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return x

# class Encoder(nn.Module):
#     def __init__(self, d_model, nhead, num_layers, dropout=0.2):
#         super().__init__()

#         self.conv1 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=3, padding=1)
        
#         self.pos_encoder = PositionalEncoding(d_model)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout, batch_first=True)
#         self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

#         self.linear1 = nn.Linear(d_model, 512)
#         self.linear2 = nn.Linear(512, 300)
#         self.dropout_layer = nn.Dropout(dropout)

#         self.layer_norm1 = nn.LayerNorm(d_model)
#         self.layer_norm2 = nn.LayerNorm(512)

#     def forward(self, x):
#         # Apply convolutional layers
#         x = x.permute(0, 2, 1)  # Change shape to (batch_size, 512, n_frames)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = x.permute(0, 2, 1)  # Change shape back to (batch_size, n_frames, d_model)
        
#         # Apply positional encoding and Transformer encoder
#         x = self.pos_encoder(x)
#         out = self.encoder(x)

#         # Apply linear layers and normalization
#         out = self.layer_norm1(out)
#         out = F.relu(self.linear1(out))
#         out = self.layer_norm2(out)
#         out = self.dropout_layer(out)
#         logits = self.linear2(out)
#         return logits



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
            
            x = torch.transpose(x, 0, 1) # changing shape of X to (batch_size, n_frames, gd_coeff)
            y = torch.transpose(y, 0, 1) # changing shape of y to (batch_size, n_frames)
            # print(f"shape of x : {x.shape}, shape of target : {y.shape}")
            #print("aaaa", torch.cuda.memory_allocated(device=device))
            x, y = x.to(device), y.to(device)
            #print("ffff", torch.cuda.memory_allocated())

            optimizer.zero_grad()
            #print("gggg", torch.cuda.memory_allocated())
            #print("aaaaa", y.min(), y.max())
            train_out = model(x)
            
            #print("hhhh", torch.cuda.memory_allocated())
            # print(f"out shape : {train_out.shape}")
            loss = criterion(train_out.reshape(-1, 300), y.flatten())
            loss.backward()
            optimizer.step()
            #print("iiii", torch.cuda.memory_allocated())
            train_step_loss += loss.item()
            _, train_correct_outputs = torch.max(train_out.reshape(-1, 300), dim=1)
            train_step_correct_pred += (train_correct_outputs == y.flatten()).sum().item()
            #print("dddd", torch.cuda.memory_allocated)

        
        training_metrics["train_avg_loss"].append(train_step_loss / len(train_loader))
        training_metrics["train_avg_acc"].append(train_step_correct_pred / len(train_loader))
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (x_val, y_val) in enumerate(valid_loader):
                x_val = torch.transpose(x_val, 0, 1)
                y_val = torch.transpose(y_val, 0, 1)
                x_val, y_val = x_val.to(device), y_val.to(device)
                valid_out = model(x_val)
                valid_loss = criterion(valid_out.reshape(-1, 300), y_val.flatten())
                valid_step_loss += valid_loss.item()
                _, valid_correct_outputs = torch.max(valid_out.reshape(-1, 300), dim=1)
                valid_step_correct_pred += (valid_correct_outputs == y_val.flatten()).sum().item()

            training_metrics["valid_avg_loss"].append(valid_step_loss / len(valid_loader))
            training_metrics["valid_avg_acc"].append(valid_step_correct_pred / len(valid_loader))
            
        # if step % 100 == 0:
     
        display(step, training_metrics)
        checkpint_store(model, training_metrics, metric, step)
        plot(dump_dir, model_name, **training_metrics)


t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r-a  # free inside reserved
print(t, r, a, f)

#print("bbbb", torch.cuda.memory_allocated())

model = Encoder(d_model=256, nhead=8, num_layers=4).to(device)
#print("cccc", torch.cuda.memory_allocated())

#model = ConvEncoder(input_dim = 512, conv_num_layers=2, kernel_sizes=[3,3], d_model=128, nhead=2, transformer_num_layers=2, dropout_rate=0.1).to(device)
#model = TemporalPyramidNetwork()
cross_entropy_loss_weights = torch.full((300,), 2.5)
cross_entropy_loss_weights[0] = 1
cross_entropy_loss_weights = cross_entropy_loss_weights.float().to(device)
#print(f"cross_entropy_loss_weights shape : {cross_entropy_loss_weights.shape}")
loss_fn = nn.CrossEntropyLoss(weight=cross_entropy_loss_weights)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
epochs = 35
#print(model)
#print("eeee", torch.cuda.memory_allocated())

dump_dir = "/speech/nishanth/clean_research/2_layer_pos_gd_1.06exps"
os.makedirs(dump_dir, exist_ok=True)
last_model_save_epoch = 0

model_name = "weighted_CEL_one_radius_adam_4_layer_pos_gd_1.06"
#     for epoch_file in epoch_files: 
#         match = epoch_pattern.match(epoch_file)
#         if match:
#             epoch_number = int(match.group(1))
#             last_model_save_epoch = max(last_model_save_epoch, epoch_number)

#     if last_model_save_epoch > 0:
#         checkpoint_path = os.path.join(os.path.join(dump_dir, "trained_models"), f"{model_name}_fake_cent_model_{last_model_save_epoch}epoch.pth")
#         model.load_state_dict(torch.load(checkpoint_path))

# pretrained_checkpoint = "/speech/nishant/roots_exps/trained_models/weighted_CEL_one_radius_step_200_loss_best.pth"
# model = torch.load(pretrained_checkpoint)  

# for t in range(last_model_save_epoch + 1, epochs):
    
#     print(f"Epoch {t}\n------------------------")
#     train_transformer(train_dataloader, model, loss_fn, optimizer, is_print=True)
#     if t%50==0:
#         valid_transformer(valid_dataloader, model, loss_fn)

#     torch.save(model.state_dict(), f"{model_save_path}/{model_name}_fake_cent_model_{t}epoch.pth")
#     if os.path.exists(f"{model_save_path}/{model_name}_fake_cent_model_{t-1}epoch.pth"):
#         os.remove(f"{model_save_path}/{model_name}_fake_cent_model_{t-1}epoch.pth")
#     last_model_save_epoch = t

train_and_eval(model, train_dataloader, valid_dataloader, optimizer, loss_fn, epochs, device, dump_dir, model_name, "valid_avg_loss", last_epoch=0)
