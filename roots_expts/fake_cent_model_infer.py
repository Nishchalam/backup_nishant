import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import re
from torch.nn.utils.rnn import pad_sequence


# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "cpu"
# )

device="cpu"

print(f"Using {device} device")


data_path = "/speech/nishant/clean_research/final_data"


def hz_to_bin(f):
    mask = np.where(f==0.0)
    cent = 1200 * np.log2((f / 10) + 1e-9)
    cent -= (1997.37 + 20)
    cent[mask] = 0.0
    bin_ = np.floor(cent / 20)
    return np.minimum(bin_, 300)


def bin_to_cent(bin_):
    zero_bin = torch.where(bin_==0)
    cent = 1997.37 + (10 * bin_)
    cent[zero_bin] = 0.0
    return cent


def mel_to_freq(bin):
    pass
    


def infer_matrices(correct, _, all_preds, tgt_hz_list, bin_range:int=4, cent_range:int=50):

    # RPA --->

    pred_prob = torch.nn.functional.softmax(all_preds, dim=-1)
    unvoiced_ground_truth = torch.where(correct==0)
    voiced_ground_truth = torch.where(correct!=0)
    pred = torch.argmax(pred_prob, dim=-1)
    voiced_pred_prob = pred_prob[voiced_ground_truth]
    voiced_pred = torch.argmax(voiced_pred_prob, dim=-1)
    correct_voiced = correct[voiced_ground_truth]
    
    
    print(f"{torch.sum(correct==0).item()}\t{torch.sum(pred==0).item()}\t{torch.sum(torch.eq((correct==0),(pred!=0))).item()}")
    
    UVE = torch.sum(torch.eq((correct==0),(pred!=0))).item() / torch.sum(correct==0).item()
    VUE = torch.sum(torch.eq((correct!=0),(pred==0))).item() / torch.sum(correct!=0).item()
    
    pred_idx_range = [] 
    for idx in voiced_pred:
        pred_idx_range.append(torch.arange(idx - bin_range, idx + bin_range + 1))
    pred_idx_range = torch.stack(pred_idx_range).to(device)

    voiced_pred_prob_range = voiced_pred_prob[torch.arange(voiced_pred_prob.shape[0]).unsqueeze(1).expand(-1, bin_range*2 + 1), pred_idx_range]
    voiced_pred_prob_range /= voiced_pred_prob_range.sum(dim=-1, keepdims=True)

    pred_cent = torch.sum(voiced_pred_prob_range * bin_to_cent(pred_idx_range), dim=-1) 
    correct_cent = bin_to_cent(correct_voiced)
    correct_cent_in_range = (torch.abs(correct_cent - pred_cent)<=cent_range).sum().item()
    print(cent_range)
    RPA = correct_cent_in_range / len(correct_cent) 
    print(f"RPA : {RPA*100}")
    print(f"UVE : {UVE}")
    print(f"VUE : {VUE}")


PAD_IDX=0
def collate_fn(batch):
    src_batch, tgt_batch, tgt_hz = [], [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(torch.tensor(np.load(src_sample), dtype=torch.float32))
        tgt_batch.append(torch.tensor(hz_to_bin(np.load(tgt_sample)), dtype=torch.long))
        tgt_hz.append(torch.tensor(np.load(tgt_sample), dtype=torch.long))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=False)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=False)
    tgt_hz = pad_sequence(tgt_hz, padding_value=PAD_IDX, batch_first=False)
    return src_batch.to(device), tgt_batch.to(device), tgt_hz.to(device)

# def collate_fn(batch):
#     src_batch, tgt_batch = [], []
#     for src_sample, tgt_sample in batch:
#         src_batch.append(torch.tensor(np.load(src_sample), dtype=torch.float32))
#         tgt_batch.append(torch.tensor(hz_to_bin(np.load(tgt_sample)), dtype=torch.long))

#     src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=False)
#     tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=False)
#     return src_batch.to(device), tgt_batch.to(device)


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.encoderlayer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoderlayer, num_layers)

        self.linear1 = nn.Linear(d_model, 512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512, 300)

    def forward(self, x):
        # expected shape of src (batch_size, n_frames, gd_coeff)
        out = self.encoder(x)

        logits = self.linear2(self.relu(self.linear1(out)))
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
    for X, y, tgt_hz in dataloader:
        tgt_hz_list.append(tgt_hz)
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
    return correct_targets, predicted_targets, all_preds, torch.cat(tgt_hz_list).flatten()
   

def get_loss_function():
    cross_entropy_loss_weights = torch.full((300,), 2)
    cross_entropy_loss_weights[0] = 1
    cross_entropy_loss_weights = cross_entropy_loss_weights.float().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=cross_entropy_loss_weights)
    return loss_fn
    


# print(f"cross_entropy_loss_weights shape : {cross_entropy_loss_weights.shape}")
# loss_fn = nn.CrossEntropyLoss()

model_path = "/speech/nishant/clean_research/trained_models/trained_models/weighted_CEL_one_radius_adam_1.002_step_25_loss_best.pth"

state_dict = torch.load(model_path).state_dict()


model = Encoder(d_model=512, nhead=2, num_layers=2).to(device)

model.load_state_dict(state_dict)

loss_fn = get_loss_function()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"number of parameters = {count_parameters(model)}")

data_path = "/speech/nishant/clean_research/final_data"

for db in [0, 5, 10, 20, "clean"]:

    file_names_0_db = os.listdir(f"/speech/nishant/roots_exps/gd/{str(db)}/test")
    print(f"processing {db} db test set")

    test_X = os.listdir(f"{data_path}/test/gd_1.002")
    test_y = os.listdir(f"{data_path}/test/labels")

    test_X = [file_name for file_name in test_X if file_name in file_names_0_db]
    test_y = [file_name for file_name in test_y if file_name in file_names_0_db]


    test_X = [os.path.join(f"{data_path}/test/gd_1.002", x) for x in test_X]
    test_y = [os.path.join(f"{data_path}/test/labels", y) for y in test_y]

    #List of tuples [ (input_link, label_link)]
    # e.g [ ('fake_gd_data/train/group_delay/1.npy', 'fake_gd_data/train/labels/1.npy') ]
    test_data = list(zip(test_X, test_y))
    batch_size = 2
    test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)
    """    - - - - - - - - - - - - - - - - - - - - - -    """

    correct, pred, all_preds, tgt_hz_list = test_transformer(test_dataloader, model, loss_fn)
    print(correct.shape, pred.shape)
    infer_matrices(correct, pred, all_preds, tgt_hz_list)
    print("\n\n")
    to_save = torch.cat((correct.unsqueeze(1), pred.unsqueeze(1)), dim=1)
    to_save = to_save.cpu().numpy()  # Convert to numpy array and move to CPU if necessary

    with open("weighted_CEL_fake_cent_infer_result_two_layer_test_0.8.txt", "wt") as f:
        for row in to_save:
            f.write("\t".join(str(e) for e in row))
            f.write("\n")


