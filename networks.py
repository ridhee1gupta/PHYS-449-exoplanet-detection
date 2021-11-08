#################################################### Nov 11, 2021 ####################################################
"""
Local Global and Combined Neural Network Architecture
"""
#################################################### Imports ####################################################
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch


#################################################### Code ####################################################

# Local View
class LocalNetwork(nn.Module):
    def __init__(self):
        super(LocalNetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, padding = 'same')
        self.conv2 = nn.Conv1d(16, 16, 5, padding='same')
        self.mp1 = nn.MaxPool1d(3,2)

        self.conv3 = nn.Conv1d(16, 32, 5, padding = 'same')
        self.conv4 = nn.Conv1d(32, 32, 5, padding='same')
        self.mp2 = nn.MaxPool1d(3,2)

    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.mp1(nn.functional.relu(z))

        z = self.conv3(z)
        z = self.conv4(z)
        z = self.mp2(nn.functional.relu(z))

        return z

class GlobalNetwork(nn.Module):
    def __init__(self):
        super(GlobalNetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, padding = 'same')
        self.conv2 = nn.Conv1d(16, 16, 5, padding='same')
        self.mp1 = nn.MaxPool1d(3,2)

        self.conv3 = nn.Conv1d(16, 32, 5, padding = 'same')
        self.conv4 = nn.Conv1d(32, 32, 5, padding='same')
        self.mp2 = nn.MaxPool1d(3,2)

        self.conv5 = nn.Conv1d(32, 64, 5, padding = 'same')
        self.conv6 = nn.Conv1d(64, 64, 5, padding='same')
        self.mp3 = nn.MaxPool1d(3,2)

        self.conv7 = nn.Conv1d(64, 128, 5, padding = 'same')
        self.conv8 = nn.Conv1d(128, 128, 5, padding='same')
        self.mp4 = nn.MaxPool1d(3,2)


    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.mp1(nn.functional.relu(z))

        z = self.conv3(z)
        z = self.conv4(z)
        z = self.mp2(nn.functional.relu(z))

        z = self.conv5(z)
        z = self.conv6(z)
        z = self.mp3(nn.functional.relu(z))

        z = self.conv7(z)
        z = self.conv8(z)
        z = self.mp4(nn.functional.relu(z))

        return z

class FullyConnectedNetwork(nn.Module):
    def __init__(self, LocalNetwork= LocalNetwork, GlobalNetwork= GlobalNetwork):
        super(FullyConnectedNetwork, self).__init__()
        self.LocalNetwork = LocalNetwork()
        self.GlobalNetwork = GlobalNetwork()
        insize = 128*42+32*12+2
        self.lin1 = nn.Linear(insize,512)
        self.lin2 = nn.Linear(512,512)
        self.lin3 = nn.Linear(512,512)
        self.lin4 = nn.Linear(512,1)
        self.out = nn.Sigmoid()

    def forward(self, x_local,x_global,impact,rad_rat):
        local_out = self.LocalNetwork(x_local)
        global_out = self.GlobalNetwork(x_global)

        z = torch.cat((torch.flatten(local_out, start_dim = 1),torch.flatten(global_out, start_dim= 1), impact, rad_rat),dim = 1)
        z = self.lin1(nn.functional.relu(z))
        z = self.lin2(nn.functional.relu(z))
        z = self.lin3(nn.functional.relu(z))
        z = self.lin4(nn.functional.relu(z))
        z = self.out(z)

        return z



def train_net(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    loss_lst = []
    for batch, (X_local, X_global, impact, rad_rat, y_vals) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X_local, X_global, impact, rad_rat)
        loss = loss_fn(pred, y_vals)
        loss_lst.append(loss)
        # Backpropagation
        optimizer.zero_grad() # zero the gradients
        loss.backward() # backpass
        optimizer.step() # step

        if batch % 100 == 0:
            loss, current = loss.item(), batch * X_local.shape[0]
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

if __name__ == "__main__":
    # Load data
    local_views = np.load('NpyData/local_view.npy')
    local_views= torch.DoubleTensor(local_views.reshape(local_views.shape[0], 1, local_views.shape[1]))
    global_views = np.load('NpyData/global_view.npy')
    global_views= torch.DoubleTensor(global_views.reshape(global_views.shape[0], 1, global_views.shape[1]))

    impact = np.load('NpyData/tce_impact.npy')
    rad_rat = np.sqrt(np.load('NpyData/tce_depth.npy'))

    labels = np.load('NpyData/av_training_set.npy')[:,1] # 67 = "Planet Candidate", 69 = "Eclipsing Binary", 74 = "Junk" # TODO: Figure out what 'R' and 'U' are
    labels = labels == 67
    labels = labels.reshape(labels.shape[0],1)

    y = torch.FloatTensor(labels)

    batch_size = 10
    lr = 0.001

    class Data(Dataset):
        def __init__(self, X_local, X_global, impact, rad_rat, y_vals):
            self.X_local = X_local
            self.X_global = X_global
            self.impact = impact
            self.rad_rat = rad_rat
            self.y = y_vals

        def __len__(self):
            return self.y.shape[0]

        def __getitem__(self, idx):
            local_vals = self.X_local[idx, :]
            global_vals = self.X_global[idx, :]
            impact_vals = self.impact[idx]
            rad_rat_vals = self.rad_rat[idx]
            labels = self.y[idx]
            return local_vals,global_vals,impact_vals,rad_rat_vals, labels

    data = Data(local_views,global_views,impact,rad_rat,y)

    dataloader = DataLoader(data,batch_size = batch_size)

    fc_network = FullyConnectedNetwork()
    fc_network.double()

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(fc_network.parameters(), lr=lr)
    train_net(dataloader, fc_network, loss, optimizer)

    x = 1