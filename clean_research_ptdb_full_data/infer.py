
import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import re
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

#device="cpu"

#print(f"Using {device} device")


data_path = "/speech/nishanth/mir_dataset/final_data_awgn"


def hz_to_bin(f):
    mask = np.where(f==0.0)
    cent = 1200 * np.log2((f / 10) + 1e-9)
    cent -= (1997.37 - 20)
    cent[mask] = 0.0
    bin_ = np.floor(cent / 20)
    return np.minimum(bin_, 300)


def bin_to_cent(bin_):
    zero_bin = torch.where(bin_==0)
    #cent = 1997.37 + (10 * bin_)
    cent = 1997.37 + ((20 * bin_) -10)
    cent[zero_bin] = 0.0
    return cent

def hz_to_cent(f):
    mask = np.where(f==0.0)
    cent = 1200 * np.log2((f / 10) + 1e-9)
    cent[mask] = 0.0
    return np.round(cent)

def cent_to_hz(cent):
    mask = torch.where(cent==0.0)
    cent = cent/1200
    return 10*(2**cent)

def bin_to_hz(bin_):
    return cent_to_hz(bin_to_cent(bin_))
    



def infer_matrices(correct, _, all_preds, bin_range: int = 4, cent_range: int = 50):
    # Apply softmax to get probabilities
    pred_prob = torch.nn.functional.softmax(all_preds, dim=-1)
    
    # Get the predicted class (argmax) for each frame
    pred = torch.argmax(pred_prob, dim=-1)

    # Determine voiced (non-zero) and unvoiced (zero) ground truth indices
    voiced_indices = correct != 0
    unvoiced_indices = correct == 0
    
    # Get predictions and ground truths for voiced frames only
    pred_voiced = pred[voiced_indices]
    correct_voiced = correct[voiced_indices]


    pred_idx_range = [] 
    for idx in pred:
        pred_idx_range.append(torch.arange(idx - bin_range, idx + bin_range + 1))
    pred_idx_range = torch.stack(pred_idx_range).to(device)

    voiced_pred_prob_range = pred_prob[torch.arange(pred_prob.shape[0]).unsqueeze(1).expand(-1, bin_range*2 + 1), pred_idx_range]
    voiced_pred_prob_range /= voiced_pred_prob_range.sum(dim=-1, keepdims=True)

    pred_cent = torch.round(torch.sum(voiced_pred_prob_range * bin_to_cent(pred_idx_range), dim=-1))
    # pred_cent = median_filter(pred_cent, 5)
    # Convert the predicted and correct bin values to cent values
    # pred_cent = bin_to_cent(pred_voiced)
    correct_cent = correct[voiced_indices]
    # pred_cent_ = bin_to_cent(pred)
    # Calculate VRR: proportion of correctly voiced predictions
    voiced_pred_indices = pred_cent != 0
    correct_voiced_pred_indices = correct != 0

    matching_voiced_indices = voiced_pred_indices & correct_voiced_pred_indices
    VRR = matching_voiced_indices.sum().item() / correct_voiced_pred_indices.sum().item()

    # Calculate RPA: percentage of correct cent values within cent_range
    correct_within_range = torch.abs(correct_cent - pred_cent[voiced_indices]) <= cent_range
    RPA = correct_within_range.sum().item() / len(correct_cent)

    # GPE: Gross Pitch Error
    pred_voiced_hz = 1 / bin_to_hz(pred_voiced)
    correct_voiced_hz = 1 / bin_to_hz(correct_voiced)
    GPE = torch.sum((torch.abs(pred_voiced_hz - correct_voiced_hz) > (10 / 8000))).item() / len(correct_voiced_hz)

    # FPE: Fine Pitch Error
    FPE = torch.sum((torch.abs(pred_voiced_hz - correct_voiced_hz) <= (10 / 8000))).item() / len(correct_voiced_hz)

    # UVE: Unvoiced to Voiced Error
    UVE = torch.sum(unvoiced_indices & (pred != 0)).item() / torch.sum(unvoiced_indices).item()

    # VUE: Voiced to Unvoiced Error
    VUE = torch.sum(voiced_indices & (pred == 0)).item() / torch.sum(voiced_indices).item()

    # Print results
    print(f"RPA: {RPA * 100:.2f}%")
    print(f"VRR: {VRR * 100:.2f}%")
    print(f"GPE: {GPE * 100:.2f}%")
    print(f"FPE: {FPE * 100:.2f}%")
    print(f"UVE: {UVE * 100:.2f}%")
    print(f"VUE: {VUE * 100:.2f}%")

    # Optional: Write cent values comparison to file (optional, can be removed if not needed)
    with open("cent_compare_hz.txt", "wt") as f:
        for c, p in zip(cent_to_hz(correct_cent), cent_to_hz(pred_cent)):
            f.write(f"{c:.3f}\t{p:.3f}\n")

    return pred_cent
        



PAD_IDX=0
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(torch.tensor(np.load(src_sample), dtype=torch.float32).squeeze(0))
        tgt_batch.append(torch.tensor(hz_to_cent(np.load(tgt_sample)), dtype=torch.long))

    # print([a.shape for a in src_batch])
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=False)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=False)
    return src_batch.to(device), tgt_batch.to(device)

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
    def __init__(self, d_model, nhead, num_layers, dropout=0.4):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same')
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=3, padding='same')
        
        # Replacing GRU with LSTM
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True)
        
        self.pos_encoder = PositionalEncoding(d_model * 2)  # d_model * 2 due to bidirectional LSTM
        self.encoder_layer = nn.TransformerEncoderLayer(d_model * 2, nhead, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.linear1 = nn.Linear(d_model * 2, 512)  # Adjusting input dimension to match LSTM output
        self.linear2 = nn.Linear(512, 300)
        self.dropout_layer = nn.Dropout(dropout)

        self.layer_norm1 = nn.LayerNorm(d_model * 2)
        self.layer_norm2 = nn.LayerNorm(512)

    def forward(self, x):
        # Apply convolutional layers
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, 512, n_frames)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, n_frames, d_model)
        
        # Apply LSTM
        x, _ = self.lstm(x)

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

def test_transformer(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    print(size, num_batches)
    model.eval()
    correct_targets = []
    predicted_targets = []
    all_preds = []
    tgt_hz_list = []
    test_loss, correct = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            X = torch.transpose(X, 0, 1) # changing shape of X to (batch_size, n_frames, gd_coeff)
            y = torch.transpose(y, 0, 1) # changing shape of y to (batch_size, n_frames)    
            X, y = X.to(device), y.to(device)
            pred = model(X) # shape of pred : (batch_size, n_frames, n_bins)
            pred_idx = torch.argmax(pred, dim=-1)
            all_preds.append(pred.reshape(-1, 300))

            correct_targets.append(y.flatten())
            predicted_targets.append(pred_idx.flatten())
            # test_loss += loss_fn(pred.reshape(-1, 300), y.flatten()).item()
            correct += (pred.argmax(2) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size * num_batches
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    
    correct_targets = torch.cat(correct_targets)
    predicted_targets = torch.cat(predicted_targets)
    # print(f"predicted_targets shape : {predicted_targets.shape}")
    all_preds = torch.cat(all_preds).view(-1, 300)
    # print(f"all_preds shape : {all_preds.shape}")
    return correct_targets, predicted_targets, all_preds

def get_loss_function():
    cross_entropy_loss_weights = torch.full((300,), 2.5)
    cross_entropy_loss_weights[0] = 1
    cross_entropy_loss_weights = cross_entropy_loss_weights.float().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=cross_entropy_loss_weights)
    return loss_fn
    
def median_filter(tensor, window_size):
    half_window = window_size // 2
    padded_tensor = torch.cat([tensor[:half_window].flip(0), tensor, tensor[-half_window:].flip(0)])
    #print("Padded tensor", padded_tensor)
    filtered_tensor = tensor.clone()

    for i in range(len(tensor)):
        window = padded_tensor[i:i + window_size]
        median_value = window.median()
        filtered_tensor[i] = median_value
    
    return filtered_tensor





model_path = "/speech/nishanth/clean_research/ptdb_full_data/mudit_4_layer_pos_gd_1.008/checkpoints/step_15_loss_latest.pth"
state_dict = torch.load(model_path).state_dict()

model = torch.load(model_path)

loss_fn = get_loss_function()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"number of parameters = {count_parameters(model)}")


for db in [0, 5, 10, 20, "clean"]:
    gd = os.listdir(os.path.join(data_path,"test",str(db),"gd_thres"))
    print(f"processing {db} db test set")

    label = os.listdir(os.path.join(data_path,"test",str(db),"labels"))

    test_X = [os.path.join(f"{data_path}/test/{db}/gd_thres", x) for x in gd]
    test_y = [os.path.join(f"{data_path}/test/{db}/labels", y) for y in label]
    test_data = list(zip(test_X, test_y))   
    batch_size = 76
    test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)
    """    - - - - - - - - - - - - - - - - - - - - - -    """

    correct, pred, all_preds = test_transformer(test_dataloader, model, loss_fn)
    # print(correct.shape, pred.shape)
    pred_cent = infer_matrices(correct, pred, all_preds)
    # print("\n\n")
    to_save = torch.cat((correct.int().unsqueeze(1), pred_cent.int().detach().unsqueeze(1)), dim=1)
    to_save = to_save.cpu().numpy()  # Convert to numpy array and move to CPU if necessary
    #to_save[:, 1] = hz_to_bin(cent_to_hz(to_save[:,1]))

    with open("new_results.txt", "wt") as f:
        for row in to_save:
            f.write("\t".join(str(e) for e in row))
            f.write("\n")

# test_X = "/speech/nishanth/clean_research/ptdb_full_data/mudit_full_final_data_1.008/test/5/gd_1.008/5_mic_M09_si1996.npy"
# test_y = "/speech/nishanth/clean_research/ptdb_full_data/mudit_full_final_data_1.008/test/5/labels/5_mic_M09_si1996.npy"
# test_data = list(zip([test_X], [test_y]))    
# batch_size = 1
# test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)
# """    - - - - - - - - - - - - - - - - - - - - - -    """

# correct, pred, all_preds = test_transformer(test_dataloader, model, loss_fn)
# # print(correct.shape, pred.shape)
# pred_cent = infer_matrices(correct, pred, all_preds)
# # print("\n\n")
# to_save = torch.cat((correct.int().unsqueeze(1), pred_cent.int().detach().unsqueeze(1)), dim=1)
# to_save = to_save.cpu().numpy()  # Convert to numpy array and move to CPU if necessary
# #to_save[:, 1] = hz_to_bin(cent_to_hz(to_save[:,1]))

# with open("new_results.txt", "wt") as f:
#     for row in to_save:
#         f.write("\t".join(str(e) for e in row))
#         f.write("\n")


