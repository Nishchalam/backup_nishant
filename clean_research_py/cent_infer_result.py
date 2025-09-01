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

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "cpu"
# )

device="cpu"

#print(f"Using {device} device")



def hz_to_bin(f):
    mask = np.where(f==0.0)
    cent = 1200 * np.log2((f / 10) + 1e-9)
    cent -= (1997.37 - 20)
    cent[mask] = 0.0
    bin_ = np.floor(cent / 20)
    return np.minimum(bin_, 300)


def hz_to_cent(f):
    mask = np.where(f==0.0)
    cent = 1200 * np.log2((f / 10) + 1e-9)
    cent[mask] = 0.0
    return np.round(cent)

def bin_to_cent(bin_):
    zero_bin = torch.where(bin_==0)
    #cent = 1997.37 + (10 * bin_)
    cent = 1997.37 + ((20 * bin_) -10)
    cent[zero_bin] = 0.0
    return cent


def cent_to_hz(cent):
    mask = torch.where(cent==0.0)
    cent = cent/1200
    return 10*(2**cent)

def bin_to_hz(bin_):
    return cent_to_hz(bin_to_cent(bin_))
    

def find_GPE_FPE(pred, correct):    #

    
    pred_max= torch.argmax(pred, dim=-1)
    
    correct_voiced = correct[torch.where(correct!=0)]
    pred_voiced = pred_max[torch.where(correct!=0)]
    
    non_zero_ = torch.sum(correct!=0).item()
    zero_ = torch.sum(correct==0).item()
    
    pred_voiced_hz = 1/bin_to_hz(pred_voiced)
    correct_voiced_hz = 1/cent_to_hz(correct_voiced)
    pred_hz= 1/bin_to_hz(pred_max)
    correct_hz = 1/cent_to_hz(correct)
    
    GPE = torch.sum((pred_voiced_hz - correct_voiced_hz) > (10/8000)).item() / non_zero_
    FPE = torch.sum((pred_voiced_hz- correct_voiced_hz) <= (10/8000)).item() / non_zero_
    
    UVE = torch.sum(torch.eq((correct==0),(pred_max!=0))).item() / zero_  #this error counts the incorrect unvoiced frames detection normalized by the total number of unvoiced frames.
    VUE = torch.sum(torch.eq((correct!=0),(pred_max==0))).item() / non_zero_   #This metric quantifies the error rate in 
    return GPE, FPE, UVE, VUE


def infer_matrices(correct, _, all_preds, bin_range:int=4, cent_range:int=50):

    # RPA --->

    pred_prob = torch.nn.functional.softmax(all_preds, dim=-1)
    unvoiced_ground_truth = torch.where(correct==0)
    voiced_ground_truth = torch.where(correct!=0)
    pred = torch.argmax(pred_prob, dim=-1)
    #print(f"pred shape : {pred.shape}")

    voiced_pred_prob = pred_prob[voiced_ground_truth]
    voiced_pred = torch.argmax(voiced_pred_prob, dim=-1)
    correct_voiced = correct[voiced_ground_truth]


    #print(f"{torch.sum(correct==0).item()}\t{torch.sum(pred==0).item()}\t{torch.sum(torch.eq((correct==0),(pred!=0))).item()}")
    

    
    pred_idx_range = [] 
    for idx in pred:
        pred_idx_range.append(torch.arange(idx - bin_range, idx + bin_range + 1))
    pred_idx_range = torch.stack(pred_idx_range).to(device)

    voiced_pred_prob_range = pred_prob[torch.arange(pred_prob.shape[0]).unsqueeze(1).expand(-1, bin_range*2 + 1), pred_idx_range]
    voiced_pred_prob_range /= voiced_pred_prob_range.sum(dim=-1, keepdims=True)

    pred_cent = torch.round(torch.sum(voiced_pred_prob_range * bin_to_cent(pred_idx_range), dim=-1))


    # pred_cent = median_filter( pred_cent, 5)
    
    pred_cent_ = pred_cent
    

    pred_cent = pred_cent[voiced_ground_truth]
    non_zero_pred_cent = (pred_cent!= 0).sum().item()
    
    


    correct_cent = correct_voiced
    #print("aaaaaaaaaaa", pred_cent.shape, correct_cent.shape)
    correct_cent_in_range = (torch.abs(correct_cent - pred_cent)<=cent_range).sum().item()
    
    not_correct_truth = correct_cent[torch.where(torch.abs(correct_cent - pred_cent)>cent_range)]
    not_correct_pred = pred_cent[torch.where(torch.abs(correct_cent - pred_cent)>cent_range)]
    #print(not_correct_truth.shape)

    with open("not_correct.txt", "w") as f:
        to_write = torch.cat((not_correct_truth.unsqueeze(1).detach(), not_correct_pred.unsqueeze(1).detach()), dim=1).cpu().numpy()
        #print(to_write)
        for line in to_write:
            f.write("\t".join(str(e) for e in line))
            f.write("\n")

    #print(cent_range)

    # print("aaa", non_zero_pred_cent, len(correct_cent))

    RPA = correct_cent_in_range / len(correct_cent) 
    VRR = non_zero_pred_cent / len(correct_cent)
    
    GPE, FPE, UVE, VUE = find_GPE_FPE(all_preds, correct)
    print(f"RPA: {RPA * 100:.2f}%")
    print(f"VRR: {VRR * 100:.2f}%")
    # print(f"GPE: {GPE * 100:.2f}%")
    # print(f"FPE: {FPE * 100:.2f}%")
    # print(f"UVE: {UVE * 100:.2f}%")
    # print(f"VUE: {VUE * 100:.2f}%")
    with open("cent_compare_hz.txt", "wt") as f:
        to_write = torch.stack(((cent_to_hz(correct_cent)), cent_to_hz(pred_cent)), ).transpose(1,0).tolist()
        # print(len(to_write))
        # print(correct_cent.shape, pred_cent.shape)
        for line in to_write:
            f.write(f"{line[0]:.3f}\t{line[1]:.3f}\n")

    return pred_cent_
        



PAD_IDX=0
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(torch.tensor(np.load(src_sample), dtype=torch.float32)[1,:,:])
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


def test_transformer(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    print(size, num_batches)
    model.eval()
    correct_targets = []
    predicted_targets = []
    all_preds = []
    tgt_hz_list = []
    correct = 0
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

    # test_loss /= num_batches
    correct /= size * num_batches
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%\n")
    
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


model_path = "/speech/nishanth/clean_research/2_layer_pos_gd_1.005exps/checkpoints/step_13_loss_latest.pth"
model = torch.load(model_path).to(device)
state_dict = torch.load(model_path).state_dict()

loss_fn = get_loss_function()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"number of parameters = {count_parameters(model)}")

data_path = "//speech/nishanth/clean_research/final_data_1.002_1.005_1.008_ptdb"

for db in [0, 5, 10, 20, "clean"]:
    gd = os.listdir(os.path.join(data_path,"test_eq",str(db),"gd_all"))
    print(f"processing {db} db test set")

    label = os.listdir(os.path.join(data_path,"test_eq",str(db),"labels"))

    test_X = [os.path.join(f"{data_path}/test_eq/{db}/gd_all", x) for x in gd]
    test_y = [os.path.join(f"{data_path}/test_eq/{db}/labels", y) for y in label]
    test_data = list(zip(test_X, test_y))
    batch_size = 2
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