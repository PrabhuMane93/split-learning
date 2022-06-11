import os
import struct
import socket
import pickle
import time
import sys
import h5py
import numpy as np

import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD

class ECG(Dataset):
    def __init__(self, train=True):
        if train:
            with h5py.File(os.path.join('train_ecg.hdf5'), 'r') as hdf:
                self.x = hdf['x_train'][:]
                self.y = hdf['y_train'][:]
        else:
            with h5py.File(os.path.join('test_ecg.hdf5'), 'r') as hdf:
                self.x = hdf['x_test'][:]
                self.y = hdf['y_test'][:]
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float), torch.tensor(self.y[idx])


batch_size = 32

train_dataset = ECG(train=True)
test_dataset = ECG(train=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

total_batch = len(train_loader)
print(total_batch)

class EcgClient(nn.Module):
    def __init__(self):
        super(EcgClient, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 7, padding=3)  # 128 x 16
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(2)  # 64 x 16
        self.conv2 = nn.Conv1d(16, 16, 5, padding=2)  # 64 x 16
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv1d(16, 16, 5, padding=2)  # 64 x 16
        self.relu3 = nn.LeakyReLU()
        self.pool3 = nn.MaxPool1d(2)  # 32 x 16
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(-1, 32 * 16)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

ecg_client = EcgClient()
ecg_client.to(device)

checkpoint = torch.load("init_weight.pth")
ecg_client.conv1.weight.data = checkpoint["conv1.weight"]
ecg_client.conv1.bias.data = checkpoint["conv1.bias"]
ecg_client.conv2.weight.data = checkpoint["conv2.weight"]
ecg_client.conv2.bias.data = checkpoint["conv2.bias"]
ecg_client.conv3.weight.data = checkpoint["conv3.weight"]
ecg_client.conv3.bias.data = checkpoint["conv3.bias"]

epoch = 10
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = Adam(ecg_client.parameters(), lr=lr)

def send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

host = 'localhost'
port = 9000
max_recv = 4096

s = socket.socket()
s.connect((host, port))

for e in range(epoch):
    print("Epoch {} - ".format(e+1), end='')
    
    for _, batch in enumerate(train_loader):
        x, label = batch
        x, label = x.to(device), label.to(device)
        optimizer.zero_grad()
        output = ecg_client(x)
        client_output = output.clone().detach().requires_grad_(True)
        msg = {
            'client_output': client_output,
            'label': label
        }
        msg = pickle.dumps(msg)
        send_msg(s, msg)
        msg = recv_msg(s)
        client_grad = pickle.loads(msg)
        output.backward(client_grad)
        optimizer.step()
            
    with torch.no_grad():  # calculate test accuracy
        for _, batch in enumerate(test_loader):
            x, label = batch
            x, label = x.to(device), label.to(device)
            client_output = ecg_client(x)
            msg = {
                'client_output': client_output,
                'label': label
            }
            msg = pickle.dumps(msg)
            send_msg(s, msg)
    
    msg = recv_msg(s)
    train_test_status = pickle.loads(msg)
    print(train_test_status)

print('Finished Training!')
print('Result is on the server side.')
