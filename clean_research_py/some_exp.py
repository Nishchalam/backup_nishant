import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import re
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
# device = torch.device("cpu")
print(f"Using {device} device")


def display(step, training_metrics):
    to_print = f"epoch : {step}\n" + "\n".join([f"{key} : {value[-1]}" for key, value in training_metrics.items()])        
    print(to_print)

@dataclass
class Transformer_config:
    n_classes = 500 # number of possible output labels
    block_size = 1200 # maximum number of frames in any audio
    n_embd = 256
    gd_coeff = 512
    nhead = 4
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 1024
    dropout = 0.2
    

class Transformer_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.n_classes, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        
        self.ln_in = nn.LayerNorm(config.gd_coeff)
        
        self.fc_in = nn.Linear(config.gd_coeff, config.n_embd)
        
        self.transformer = nn.Transformer(d_model=config.n_embd,
                                        nhead=config.nhead,
                                        num_encoder_layers=config.num_encoder_layers,
                                        num_decoder_layers=config.num_decoder_layers,
                                        dim_feedforward=config.dim_feedforward,
                                        dropout=config.dropout,
                                        batch_first = True)
        self.fc_out = nn.Linear(config.n_embd, config.n_classes)
        
    
    def forward(self, src, tgt):
        src = self.ln_in(src)
        src = self.fc_in(src)
        tgt_tok_embd = self.wte(tgt)
        tgt_pos = torch.arange(0, tgt.shape[1], dtype=torch.long, device = tgt.device)
        tgt_pos_embd = self.wpe(tgt_pos)
        tgt = tgt_tok_embd + tgt_pos_embd
        out = self.transformer(src, tgt)
        out = self.fc_out(out)
        return out # shape : batch_size, n_frames, n_classes
                


config = Transformer_config()
model = Transformer_model(config).to(device)
print(model)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)



training_metrics = {"train_avg_loss":[],
                    "valid_avg_loss":[]}
correct = []
pred = []

for step in range(3):
    train_step_loss = 0
    train_step_correct_pred = 0
    valid_step_loss = 0
    valid_step_correct_pred = 0
    model.train()
    
    for _ in range(100):
        
        n_frames = torch.randint(500, 900, (1,))
        x = torch.randn(32, n_frames, 512)
        y = torch.randint(0, 500, (32, n_frames))
        
        # print(f"shape of x : {x.shape}, shape of target : {y.shape}")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        train_out = model(x, y)
        # print(f"out shape : {train_out.shape}")
        loss = loss_fn(train_out.reshape(-1, config.n_classes), y.flatten())
        loss.backward()
        optimizer.step()
        _, pred_idx = torch.max(train_out.reshape(-1, 500), dim=1)
        pred.append(pred_idx)
        correct.append(y.flatten())
        train_step_loss += loss.item()
        _, train_correct_outputs = torch.max(train_out.reshape(-1, 500), dim=1)
        train_step_correct_pred += (train_correct_outputs == y.flatten()).sum().item()
    
    training_metrics["train_avg_loss"].append(train_step_loss / 100)
    # training_metrics["train_avg_acc"].append(train_step_correct_pred / ((batch_idx + 1) * batch_size))

    model.eval()
    with torch.no_grad():
        n_frames = torch.randint(500, 900, (1,))
        x_val = torch.randn(32, n_frames, 512)
        y_val = torch.randint(0, 500, (32, n_frames))
        x_val, y_val = x_val.to(device), y_val.to(device)
        valid_out = model(x_val, y_val)
        valid_loss = loss_fn(valid_out.reshape(-1, 500), y_val.flatten())
        valid_step_loss += valid_loss.item()
        _, valid_correct_outputs = torch.max(valid_out.reshape(-1, 500), dim=1)
        valid_step_correct_pred += (valid_correct_outputs == y_val.flatten()).sum().item()

        training_metrics["valid_avg_loss"].append(valid_step_loss)
        
    # if step % 100 == 0:
    display(step, training_metrics)
    
pred = torch.cat(pred).detach().cpu().numpy()
correct = torch.cat(correct).detach().cpu().numpy()

to_save = np.stack((pred, correct), axis=1)
with open("rand_result.txt", "wt") as f:
    for row in to_save:
        f.write("\t".join(str(e) for e in row))
        f.write("\n")


# state_dict = model.state_dict()
# x_attn_weight = state_dict["transformer.decoder.layers.0.multihead_attn.in_proj_weight"]

# in_layer = torch.Linear(256, 768)
# in_layer.weight = x_attn_weight

# x = torch.randn(1, 700, 512)
# y = torch.randint(1, 700)

