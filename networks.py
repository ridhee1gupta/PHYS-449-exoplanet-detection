#################################################### Nov 11, 2021 ####################################################
"""
Local Global and Combined Neural Network Architecture
"""
#################################################### Imports ####################################################
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch
from torchsampler import ImbalancedDatasetSampler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

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
        loss = loss_fn(pred.double(), y_vals.double())
        loss_lst.append(loss.item())
        # Backpropagation
        optimizer.zero_grad() # zero the gradients
        loss.backward() # backpass
        optimizer.step() # step

        if batch % 20 == 0:
            print("")
            unique, counts = np.unique(y_vals,return_counts = True)
            d = dict(zip(unique,counts))
            print("Unique Counts Dict:", d)
            # loss_avg = np.average(loss_lst[-100:])
            loss, current = loss.item(), batch * X_local.shape[0]
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss_lst

if __name__ == "__main__":
    # Load data
    local_views = np.load('NpyData/local_view.npy')
    global_views = np.load('NpyData/global_view.npy')

    impact = np.load('NpyData/tce_impact.npy')
    depth = np.load('NpyData/tce_depth.npy')

    labels = np.load('NpyData/av_training_set.npy')[:,1] # 67 = "Planet Candidate", 69 = "Eclipsing Binary", 74 = "Junk" # TODO: Figure out what 'R' and 'U' are
    labels = (labels == 67)
    labels = labels.reshape(labels.shape[0],1)

    local_views_l = local_views.tolist()
    global_views_l = global_views.tolist()
    impact_l = impact.tolist()
    depth_l = depth.tolist()
    labels_l = labels.tolist()

    pos_idxs = np.where(labels[:,0])[0]
    for idx in pos_idxs:
        for n in range(26):
            local_views_l.append(local_views_l[idx])
            global_views_l.append(global_views_l[idx])
            impact_l.append(impact_l[idx])
            depth_l.append(depth_l[idx])
            labels_l.append(labels_l[idx])
            # local_views = np.insert(local_views, -1, local_views[idx], axis = 0)
            # global_views = np.insert (global_views,-1,global_views[idx], axis = 0)
            # impact = np.insert(impact, -1,impact[idx], axis = 0)
            # depth = np.insert(depth, -1, depth[idx],axis = 0)
            # labels = np.insert(labels, -1,  labels[idx],axis = 0)

    impact = np.array(impact_l)
    depth = np.array(depth_l)
    local_views = np.array(local_views_l)
    local_views = torch.DoubleTensor(local_views.reshape(local_views.shape[0], 1, local_views.shape[1]))
    global_views = np.array(global_views_l)
    global_views = torch.DoubleTensor(global_views.reshape(global_views.shape[0], 1, global_views.shape[1]))

    labels = np.array(labels_l)


    y = torch.FloatTensor(labels)

    batch_size = 64
    lr = 0.0001

    class Data(Dataset):
        def __init__(self, X_local, X_global, impact, depth, y_vals):
            self.X_local = X_local
            self.X_global = X_global
            self.impact = impact
            self.depth = depth
            self.y = y_vals

        def __len__(self):
            return self.y.shape[0]

        def __getitem__(self, idx):
            local_vals = self.X_local[idx, :]
            global_vals = self.X_global[idx, :]
            impact_vals = self.impact[idx]
            depth_vals = self.depth[idx]
            labels = self.y[idx]
            return local_vals,global_vals,impact_vals,depth_vals, labels

    data = Data(local_views,global_views,impact,depth,y)

    dataloader = DataLoader(data,batch_size = batch_size, shuffle = True)

    fc_network = FullyConnectedNetwork()
    fc_network.double()

    # loss = nn.CrossEntropyLoss()
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(fc_network.parameters(), lr=lr)
    loss_lst = train_net(dataloader, fc_network, loss, optimizer)

    # FOR TESTING
    test_dataloader = DataLoader(data, batch_size=1000, shuffle=False)
    for batch, (X_local, X_global, impact, rad_rat, y_vals) in enumerate(test_dataloader):
        # Compute prediction and loss
        pred = fc_network(X_local, X_global, impact, rad_rat)
        pred_arr = pred.detach().numpy().reshape(pred.shape[0])
        y_arr = y_vals.detach().numpy().reshape(y_vals.shape[0])
        print(classification_report(y_arr,np.round(pred_arr)))
        break

    x = 1