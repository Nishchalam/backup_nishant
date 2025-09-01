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

train_X = os.listdir(f"{data_path}/train/gd_1.01_1.03_1.06")
train_X = [os.path.join(f"{data_path}/train/gd_1.01_1.03_1.06", x) for x in train_X]
train_y = os.listdir(f"{data_path}/train/labels")
train_y = [os.path.join(f"{data_path}/train/labels", y) for y in train_y]

valid_X = os.listdir(f"{data_path}/valid/gd_1.01_1.03_1.06")
valid_X = [os.path.join(f"{data_path}/valid/gd_1.01_1.03_1.06", x) for x in valid_X]
valid_y = os.listdir(f"{data_path}/valid/labels")
valid_y = [os.path.join(f"{data_path}/valid/labels", y) for y in valid_y]

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
    
    # Determine the maximum length for padding
    max_len = max([torch.tensor(np.load(src_sample)).shape[1] for src_sample, _ in batch])

    for src_sample, tgt_sample in batch:
        src_tensor = torch.tensor(np.load(src_sample), dtype=torch.float32)  # Shape: (3, nframes, 512)
        tgt_tensor = torch.tensor(hz_to_bin(np.load(tgt_sample)), dtype=torch.long)  # Shape: (nframes, )

        # Padding src_tensor to max_len
        if src_tensor.shape[1] < max_len:
            padding = (0, 0, 0, max_len - src_tensor.shape[1])  # Pad the second dimension (nframes)
            src_tensor = F.pad(src_tensor, padding, value=PAD_IDX)
        
        src_batch.append(src_tensor)
        tgt_batch.append(tgt_tensor)

    # Stack the padded tensors
    src_batch = torch.stack(src_batch)  # Now should be (batch_size, 3, max_len, 512)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)  # Shape: (batch_size, max_len)

    return src_batch.to(device), tgt_batch.to(device)


batch_size = 16
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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]  # Match the positional encoding with sequence length
        return x

class Encoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        # 1D Convolutions to focus on temporal dynamics
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 512), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=d_model, kernel_size=(1, 1), padding=(0, 0))
        
        # Bi-directional GRU to capture temporal dependencies
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=2, batch_first=True, dropout=dropout, bidirectional=True)

        # Self-attention mechanism
        self.attn = nn.MultiheadAttention(d_model * 2, nhead, dropout=dropout, batch_first=True)
        
        # Positional Encoding for Transformer
        self.pos_encoder = PositionalEncoding(d_model * 2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model * 2, nhead, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        # Fully connected layers
        self.fc1 = nn.Linear(d_model * 2, 256)
        self.fc2 = nn.Linear(256, 300)
        self.dropout = nn.Dropout(dropout)
        
        # Normalization layers
        self.layer_norm1 = nn.LayerNorm(d_model * 2)
        self.layer_norm2 = nn.LayerNorm(256)

    def forward(self, x):
        # Expected input shape: (3, 16, 987, 512)
        x = x.permute(1, 0, 2, 3)  # Change to (16, 3, 987, 512) for Conv2d

        # Apply Conv2d layers
        x = F.relu(self.conv1(x))  # Shape: (16, 64, 987, 1)
        x = F.relu(self.conv2(x))  # Shape: (16, d_model, 987, 1)
        x = x.squeeze(-1)          # Shape: (16, d_model, 987)
        x = x.permute(0, 2, 1)     # Shape: (16, 987, d_model)
        
        # Pass through GRU
        x, _ = self.gru(x)         # Shape: (16, 987, d_model * 2)
        
        # Apply self-attention
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output  # Add residual connection
        
        # Positional Encoding and Transformer Encoder
        x = self.pos_encoder(x)
        x = self.encoder(x)
        
        # Normalize and fully connected layers
        x = self.layer_norm1(x)
        x = F.relu(self.fc1(x))
        x = self.layer_norm2(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
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
#     def __init__(self, d_model=128, nhead=4, num_layers=2, dropout=0.1):
#         super().__init__()

#         # Adjusted Conv2d layers to handle input dimension of 512
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 512), padding=(0, 0))
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=d_model, kernel_size=(1, 1), padding=(0, 0))

#         self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=2, batch_first=True, dropout=dropout, bidirectional=True)

#         self.pos_encoder = PositionalEncoding(d_model * 2)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model * 2, nhead, dim_feedforward=256, dropout=dropout, batch_first=True)
#         self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

#         self.linear1 = nn.Linear(d_model * 2, 256)
#         self.linear2 = nn.Linear(256, 300)
#         self.dropout_layer = nn.Dropout(dropout)

#         self.layer_norm1 = nn.LayerNorm(d_model * 2)
#         self.layer_norm2 = nn.LayerNorm(256)

#     def forward(self, x):
#         #print("Input shape:", x.shape)  # (3, no. of frames, 512)

#         # Add a batch dimension to x: shape (batch_size, 3, no. of frames, 512)
#         # x = x.unsqueeze(0)

#         # Change shape to (batch_size, 3, no. of frames, 512)
#         # print("After unsqueeze:", x.shape)

#         x = x.permute(1,0,2,3)  # Change shape to (batch_size, 3, no. of frames, 512)
#         # print("After permute (for Conv):", x.shape)

#         x = F.relu(self.conv1(x))
#         #print("After Conv1:", x.shape)  # Expected: (batch_size, 64, no. of frames, 1)

#         x = F.relu(self.conv2(x))
#         #print("After Conv2:", x.shape)  # Expected: (batch_size, d_model, no. of frames, 1)

#         x = x.squeeze(-1)  # Remove the last dimension (which should be 1 after Conv2)
#         #print("After squeeze:", x.shape)  # Expected: (batch_size, d_model, no. of frames)

#         x = x.permute(0, 2, 1)  # Change shape to (batch_size, no. of frames, d_model)
#         #print("After permute:", x.shape)

#         x, _ = self.gru(x)
#         #print("After GRU:", x.shape)  # Expected: (batch_size, no. of frames, d_model * 2)

#         x = self.pos_encoder(x)
#         #print("After Positional Encoding:", x.shape)

#         out = self.encoder(x)
#         #print("After Transformer Encoder:", out.shape)

#         out = self.layer_norm1(out)
#         #print("After LayerNorm1:", out.shape)

#         out = F.relu(self.linear1(out))
#         #print("After Linear1:", out.shape)

#         out = self.layer_norm2(out)
#         #print("After LayerNorm2:", out.shape)

#         out = self.dropout_layer(out)
#         #print("After Dropout:", out.shape)

#         logits = self.linear2(out)
#         #print("Final output (logits) shape:", logits.shape)

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
epochs = 100
#print(model)
#print("eeee", torch.cuda.memory_allocated())

dump_dir = "/speech/nishanth/clean_research/2_layer_pos_gd_1.01_1.03_1.06_exps"
os.makedirs(dump_dir, exist_ok=True)
last_model_save_epoch = 0

model_name = "weighted_CEL_one_radius_adam_4_layer_pos_gd_1.01_1.03_1.06"


train_and_eval(model, train_dataloader, valid_dataloader, optimizer, loss_fn, epochs, device, dump_dir, model_name, "valid_avg_loss", last_epoch=0)
