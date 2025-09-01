
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


data_path = "/speech/nishanth/raw_exps/full_final_data_reversed_16khz"


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


def infer_matrices(correct, _, all_preds, bin_range:int=4, cent_range:int=20):

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
    def __init__(self, model_capacity="full"):
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

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_frames = np.load(src_sample)  # Load the entire file
        tgt_frames = hz_to_bin(np.load(tgt_sample))  # Convert target frequencies to bins
        
        for i in range(src_frames.shape[0]):  # Iterate over each frame
            src_batch.append(torch.tensor(src_frames[i], dtype=torch.float32))
            tgt_batch.append(torch.tensor(tgt_frames[i], dtype=torch.long))
    
    src_batch = torch.stack(src_batch).to(device)  # Stack frames into a batch
    tgt_batch = torch.stack(tgt_batch).to(device)
    
    return src_batch, tgt_batch

# def collate_fn(batch):
#     src_batch, tgt_batch = [], []
#     for src_sample, tgt_sample in batch:
#         src_batch.append(torch.tensor(np.load(src_sample), dtype=torch.float32))
#         tgt_batch.append(torch.tensor(hz_to_bin(np.load(tgt_sample)), dtype=torch.long))
#         # n_frames = torch.randint(100,900,(1,))
#         # # print(n_frames)
#         # src_batch.append(torch.randn(n_frames, 1024))
#         # tgt_batch.append(torch.randint(0,300,(n_frames, )))
#     # print([(x.shape, y.shape) for x, y in zip(src_batch, tgt_batch)])
#     src_batch = torch.cat(src_batch, dim=0).to(device)
#     tgt_batch = torch.cat(tgt_batch,dim=0).to(device)
#     # print(src_batch.shape, tgt_batch.shape)
#     return src_batch, tgt_batch


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
            # X = torch.transpose(X, 0, 1) # changing shape of X to (batch_size, n_frames, gd_coeff)
            # y = torch.transpose(y, 0, 1) # changing shape of y to (batch_size, n_frames)    
            # X, y = X.to(device), y.to(device)
            X, y = X.squeeze(0).to(device), y.squeeze(0).to(device)
            pred = model(X) # shape of pred : (batch_size, n_frames, n_bins)

            pred_idx = torch.argmax(pred, dim=-1)
            all_preds.append(pred.reshape(-1, 300))

            correct_targets.append(y.flatten())
            predicted_targets.append(pred_idx.flatten())
            test_loss += loss_fn(pred.reshape(-1, 300), y.flatten()).item()
            correct += (pred.argmax(-1) == y).type(torch.float).sum().item()

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

model_path = "/speech/nishanth/raw_exps/dump_crepe/checkpoints/step_2_loss_latest.pth"
model = torch.load(model_path).to(device)
state_dict = torch.load(model_path).state_dict()


# model = Encoder(d_model=512, nhead=2, num_layers=1).to(device)

# model.load_state_dict(state_dict)

loss_fn = get_loss_function()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"number of parameters = {count_parameters(model)}")

data_path = "/speech/nishanth/raw_exps/full_final_data_reversed_16khz"

for db in [0, 5, 10, 20, "clean"]:
    gd = os.listdir(os.path.join(data_path,"test",str(db),"raw"))
    print(f"processing {db} db test set")

    label = os.listdir(os.path.join(data_path,"test",str(db),"labels"))

    test_X = [os.path.join(f"{data_path}/test/{db}/raw", x) for x in gd]
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




