
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


data_path = "/speech/nishanth/clean_research/final_data_1.01_1.03_1.06_ptdb"


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


def cent_to_hz(cent):
    mask = torch.where(cent==0.0)
    cent /= 1200
    return 10*(2**cent)

def bin_to_hz(bin_):
    return cent_to_hz(bin_to_cent(bin_))
    

def find_GPE_FPE(pred, correct):    #
    # shape of correct (n_samples)
    # shape of pred (n_samples, n_classes)
    
    pred_max= torch.argmax(pred, dim=-1)
    
    correct_voiced = correct[torch.where(correct!=0)]
    pred_voiced = pred_max[torch.where(correct!=0)]
    
    non_zero_ = torch.sum(correct!=0).item()
    zero_ = torch.sum(correct==0).item()
    
    pred_voiced_hz = 1/bin_to_hz(pred_voiced)
    correct_voiced_hz = 1/bin_to_hz(correct_voiced)
    pred_hz= 1/bin_to_hz(pred_max)
    correct_hz = 1/bin_to_hz(correct)
    
    GPE = torch.sum((pred_voiced_hz - correct_voiced_hz) > (10/8000)).item() / non_zero_
    FPE = torch.sum((pred_voiced_hz- correct_voiced_hz) <= (10/8000)).item() / non_zero_
    
    UVE = torch.sum(torch.eq((correct==0),(pred_max!=0))).item() / zero_  #this error counts the incorrect unvoiced frames detection normalized by the total number of unvoiced frames.
    VUE = torch.sum(torch.eq((correct!=0),(pred_max==0))).item() / non_zero_   #This metric quantifies the error rate in incorrectly detecting the voiced frames
    
    # correct_voiced_cent = bin_to_cent(correct_voiced)
    # pred_voiced_cent = bin_to_cent(pred_voiced)
    
    # pred_idx_range = [] 
    # for idx in voiced_pred:
    #     pred_idx_range.append(torch.arange(idx - bin_range, idx + bin_range + 1))
    # pred_idx_range = torch.stack(pred_idx_range).to(device)
    #print(GPE, FPE, UVE, VUE)
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

    pred_cent = torch.sum(voiced_pred_prob_range * bin_to_cent(pred_idx_range), dim=-1)
    #pred_cent = median_filter( pred_cent, 3)

    pred_cent_ = pred_cent
    pred_cent = pred_cent[voiced_ground_truth]
    non_zero_pred_cent = (pred_cent!= 0).sum().item()
    

    correct_cent = bin_to_cent(correct_voiced)
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
    RPA = correct_cent_in_range / len(correct_cent) 
    VRR = non_zero_pred_cent / len(correct_cent)
    print(f"RPA : {RPA*100}")
    print(f"VRR : {VRR*100}")
    GPE, FPE, UVE, VUE = find_GPE_FPE(all_preds, correct)
    print(f"GPE : {GPE}")
    print(f"FPE : {FPE}")
    print(f"UVE : {UVE}")
    print(f"VUE : {VUE}")
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
    def __init__(self, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        # Adjusted Conv2d layers to handle input dimension of 512
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 512), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=d_model, kernel_size=(1, 1), padding=(0, 0))

        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=2, batch_first=True, dropout=dropout, bidirectional=True)

        self.pos_encoder = PositionalEncoding(d_model * 2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model * 2, nhead, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.linear1 = nn.Linear(d_model * 2, 256)
        self.linear2 = nn.Linear(256, 300)
        self.dropout_layer = nn.Dropout(dropout)

        self.layer_norm1 = nn.LayerNorm(d_model * 2)
        self.layer_norm2 = nn.LayerNorm(256)

    def forward(self, x):
        #print("Input shape:", x.shape)  # (3, no. of frames, 512)

        # Add a batch dimension to x: shape (batch_size, 3, no. of frames, 512)
        # x = x.unsqueeze(0)

        # Change shape to (batch_size, 3, no. of frames, 512)
        # print("After unsqueeze:", x.shape)

        x = x.permute(1,0,2,3)  # Change shape to (batch_size, 3, no. of frames, 512)
        # print("After permute (for Conv):", x.shape)

        x = F.relu(self.conv1(x))
        #print("After Conv1:", x.shape)  # Expected: (batch_size, 64, no. of frames, 1)

        x = F.relu(self.conv2(x))
        #print("After Conv2:", x.shape)  # Expected: (batch_size, d_model, no. of frames, 1)

        x = x.squeeze(-1)  # Remove the last dimension (which should be 1 after Conv2)
        #print("After squeeze:", x.shape)  # Expected: (batch_size, d_model, no. of frames)

        x = x.permute(0, 2, 1)  # Change shape to (batch_size, no. of frames, d_model)
        #print("After permute:", x.shape)

        x, _ = self.gru(x)
        #print("After GRU:", x.shape)  # Expected: (batch_size, no. of frames, d_model * 2)

        x = self.pos_encoder(x)
        #print("After Positional Encoding:", x.shape)

        out = self.encoder(x)
        #print("After Transformer Encoder:", out.shape)

        out = self.layer_norm1(out)
        #print("After LayerNorm1:", out.shape)

        out = F.relu(self.linear1(out))
        #print("After Linear1:", out.shape)

        out = self.layer_norm2(out)
        #print("After LayerNorm2:", out.shape)

        out = self.dropout_layer(out)
        #print("After Dropout:", out.shape)

        logits = self.linear2(out)
        #print("Final output (logits) shape:", logits.shape)

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
            test_loss += loss_fn(pred.reshape(-1, 300), y.flatten()).item()
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




# print(f"cross_entropy_loss_weights shape : {cross_entropy_loss_weights.shape}")
# loss_fn = nn.CrossEntropyLoss()

model_path = "/speech/nishanth/clean_research/2_layer_pos_gd_1.01_1.03_1.06_exps/checkpoints/step_5_loss_latest.pth"
model = torch.load(model_path).to(device)
state_dict = torch.load(model_path).state_dict()


# model = Encoder(d_model=512, nhead=2, num_layers=1).to(device)

# model.load_state_dict(state_dict)

loss_fn = get_loss_function()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"number of parameters = {count_parameters(model)}")


for db in [0, 5, 10, 20, "clean"]:
    gd = os.listdir(os.path.join(data_path,"test",str(db),"gd_1.01_1.03_1.06"))
    print(f"processing {db} db test set")

    label = os.listdir(os.path.join(data_path,"test",str(db),"labels"))

    test_X = [os.path.join(f"{data_path}/test/{db}/gd_1.01_1.03_1.06", x) for x in gd]
    test_y = [os.path.join(f"{data_path}/test/{db}/labels", y) for y in label]
    test_data = list(zip(test_X, test_y))
    batch_size = 2
    test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)
    """    - - - - - - - - - - - - - - - - - - - - - -    """

    correct, pred, all_preds = test_transformer(test_dataloader, model, loss_fn)
    # print(correct.shape, pred.shape)
    pred_cent = infer_matrices(correct, pred, all_preds)
    # print("\n\n")
    to_save = torch.cat((bin_to_cent(correct).int().unsqueeze(1), pred_cent.int().detach().unsqueeze(1)), dim=1)
    to_save = to_save.cpu().numpy()  # Convert to numpy array and move to CPU if necessary
    #to_save[:, 1] = hz_to_bin(cent_to_hz(to_save[:,1]))

    with open("new_results.txt", "wt") as f:
        for row in to_save:
            f.write("\t".join(str(e) for e in row))
            f.write("\n")




