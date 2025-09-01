import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import re
from torch.nn.utils.rnn import pad_sequence


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")


data_path = "/speech/nishanth/roots_exps/final_data"

train_X = os.listdir(f"{data_path}/train/group_delay")
train_X = [os.path.join(f"{data_path}/train/group_delay", x) for x in train_X]
train_y = os.listdir(f"{data_path}/train/labels")
train_y = [os.path.join(f"{data_path}/train/labels", y) for y in train_y]

test_X = os.listdir(f"{data_path}/test/group_delay")
test_X = [os.path.join(f"{data_path}/test/group_delay", x) for x in test_X]
test_y = os.listdir(f"{data_path}/test/labels")
test_y = [os.path.join(f"{data_path}/test/labels", y) for y in test_y]

#List of tuples [ (input_link, label_link)]
# e.g [ ('fake_gd_data/train/group_delay/1.npy', 'fake_gd_data/train/labels/1.npy') ]
train_data = list(zip(train_X, train_y))
test_data = list(zip(test_X, test_y))

"""    - - - - - - - - - - - - - - - - - - - - - -    """

def hz_to_bin(f):
    mask = np.where(f==0.0)
    
    cent = 1200 * np.log2((f / 10) + 1e-9)
    cent -= (1997.37 + 20)
    cent[mask] = 0.0
    bin_ = np.floor(cent / 20)

    return np.minimum(bin_, 300)


PAD_IDX=0
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(torch.tensor(np.load(src_sample), dtype=torch.float32).transpose(0,1).reshape(-1, 3*512))
        tgt_batch.append(torch.tensor(hz_to_bin(np.load(tgt_sample)), dtype=torch.long))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=False)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=False)
    return src_batch.to(device), tgt_batch.to(device)

batch_size = 2
train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)

# for X,y in train_dataloader:
#     print(f"Shape of X[N_frames<Batch_size<GD_coeff]:{X.shape}")
#     print(f"Shape of y:{y.shape}{y.dtype}")
#     break

# for X,y in test_dataloader:
#     print(f"Shape of X[N_frames<Batch_size<GD_coeff]:{X.shape}")
#     print(f"Shape of y:{y.shape}{y.dtype}")
#     break


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lstm = nn.LSTM(1536,512)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(512,300)
        self.softy = nn.Softmax(-1)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        logits = self.linear(self.relu(x))
        return self.softy(logits)


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


# model = NeuralNetwork().to(device)
#print(model)


# fake_input, fake_label = collate_fn([train_data[0]])
# print(fake_input.shape, fake_label.shape)
# model_pred = model(fake_input).squeeze()
# print(model_pred.shape)

# def corss_entropy_loss(pred, label):
#     # shape of pred 


def train(dataloader, model, loss_fn, optimizer, is_print=False):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred.reshape(-1, 300), y.flatten())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 ==0:
            loss, current = loss.item(), (batch+1)*len(X)
            if is_print:
                print(f"loss:{loss:>7f} [{current/size}]")



def train_transformer(dataloader, model, loss_fn, optimizer, is_print=False):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = torch.transpose(X, 0, 1) # changing shape of X to (batch_size, n_frames, gd_coeff)
        y = torch.transpose(y, 0, 1) # changing shape of y to (batch_size, n_frames)
        # print(f"shape of x : {X.shape}, shape of target : {y.shape}")
        X, y = X.to(device), y.to(device)
        pred = model(X)
        # print(f"batch : {batch} of {len(dataloader)}")
        # print(f"shape of pred and y in train : {pred.shape}, {y.shape}")
        # print(f"shape of pred : {pred.shape}")
        loss = loss_fn(pred.reshape(-1, 300), y.flatten())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 ==0:
            loss, current = loss.item(), (batch+1)*len(X)
            if is_print:
                print(f"loss:{loss:>7f} [{current/size}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    for X, y in dataloader:
        X,y = X.to(device), y.to(device)
        pred = model(X)
        test_loss +=loss_fn(pred.reshape(-1,300), y.flatten()).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")




def test_transformer(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    for X, y in dataloader:
        X = torch.transpose(X, 0, 1) # changing shape of X to (batch_size, n_frames, gd_coeff)
        y = torch.transpose(y, 0, 1) # changing shape of y to (batch_size, n_frames)
        # print(f"shape of x : {X.shape}, shape of target : {y.shape}")
        X,y = X.to(device), y.to(device)
        pred = model(X)
        # print(f"shape of pred and y in test : {pred.reshape(-1, 300).shape}, {y.flatten().shape}")
        test_loss +=loss_fn(pred.reshape(-1, 300), y.flatten()).item()
        correct += (pred.argmax(2) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")



model = Encoder(d_model=1536, nhead=2, num_layers=2).to(device)
cross_entropy_loss_weights = torch.full((300,), 2)
cross_entropy_loss_weights[0] = 0.8
cross_entropy_loss_weights = cross_entropy_loss_weights.float()
print(f"cross_entropy_loss_weights shape : {cross_entropy_loss_weights.shape}")
loss_fn = nn.CrossEntropyLoss(weight=cross_entropy_loss_weights)
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4)
epochs = 500
print(model)
model_save_path = "/speech/nishanth/roots_exps/trained_models"
last_model_save_epoch = 0

model_name = "weighted_CEL"
pretrained = False # set true if continuing training on a trained model
if pretrained:
    epoch_files = os.listdir(model_save_path)
    epoch_pattern = re.compile(r"{model_name}_fake_cent_model_(\d+)epoch.pth")

    for epoch_file in epoch_files:
        match = epoch_pattern.match(epoch_file)
        if match:
            epoch_number = int(match.group(1))
            last_model_save_epoch = max(last_model_save_epoch, epoch_number)

    if last_model_save_epoch > 0:
        checkpoint_path = os.path.join(model_save_path, f"{model_name}_fake_cent_model_{last_model_save_epoch}epoch.pth")
        model.load_state_dict(torch.load(checkpoint_path))

    

for t in range(last_model_save_epoch + 1, epochs):
    
    print(f"Epoch {t}\n------------------------")
    train_transformer(train_dataloader, model, loss_fn, optimizer, is_print=True)
    if t%50==0:
        test_transformer(test_dataloader, model, loss_fn)

    torch.save(model.state_dict(), f"{model_save_path}/{model_name}_fake_cent_model_{t}epoch.pth")
    if os.path.exists(f"{model_save_path}/{model_name}_fake_cent_model_{t-1}epoch.pth"):
        os.remove(f"{model_save_path}/{model_name}_fake_cent_model_{t-1}epoch.pth")
    last_model_save_epoch = t
