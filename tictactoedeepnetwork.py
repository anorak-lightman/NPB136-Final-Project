import os
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np

class TicTacToeDeepNet(nn.Module):
    def __init__(self):
        super(TicTacToeDeepNet, self).__init__()
        self.fc1 = nn.Linear(9, 15)
        self.fc2 = nn.Linear(15, 6)
        self.fc3 = nn.Linear(6, 1)
        #self.fc4 = nn.Linear(8, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
      #  x = F.relu(x)

      #  x = self.fc4(x)
        return x
    
class TicTacToeDataset(Dataset):
    def __init__(self, rawdata):
        rawlabels = rawdata[:,9:]
        rawdatabase = rawdata[:,:9]
        self.labels = torch.from_numpy(rawlabels)
        self.database = torch.from_numpy(rawdatabase)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = self.database[idx]
        label = self.labels[idx]
        return data.float(), label.float()
    

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = lossfn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
           print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

def test():
    network.eval()

    test_loss = 0
    correct = 0
    size = len(test_loader.dataset)
    num_batches = len(test_loader)

    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += lossfn(output, target).item()
            correct += (torch.round(output) == target).type(torch.float).sum().item()

    test_loss /= num_batches
    averagecorrect = correct / size
    print(f"Test Error: \n NumCorrect: {(int(correct))}/{(size)}, Accuracy: {(100*averagecorrect):>0.1f}%, Avg loss: {test_loss:>8f} \n")

batch_size_train = 10
batch_size_test = 100

training_data_file = np.loadtxt('/Users/davidroth/Documents/NPB136/Final Project/tictac_single.txt', dtype = float, max_rows = 5551)

test_data_file = np.loadtxt('/Users/davidroth/Documents/NPB136/Final Project/tictac_single.txt', dtype = float, skiprows = 5551)

training_data = TicTacToeDataset(training_data_file)

test_data = TicTacToeDataset(test_data_file)

train_loader = torch.utils.data.DataLoader(training_data, batch_size = batch_size_train, shuffle = True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size_test, shuffle = True)

learning_rate = 0.01
momentum = 0.5

network = TicTacToeDeepNet()

optimizer = optim.SGD(network.parameters(), lr = learning_rate, momentum = momentum)

lossfn = nn.MSELoss()

epochs = 10
test()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(t + 1)
    test()
print("Done!")
