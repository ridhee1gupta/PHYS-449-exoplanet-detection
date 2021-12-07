#################################################### Purpose ####################################################
"""
Local Global and Combined Neural Network Architecture
"""
#################################################### Imports ####################################################
import numpy as np
import torch.nn as nn
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



def train_net(dataloader, model, loss_fn, optimizer, num_iters):
    size = len(dataloader.dataset)
    loss_lst = []
    n = 0

    while n < num_iters:
        for batch, (X_local, X_global, impact, rad_rat, y_vals) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X_local, X_global, impact, rad_rat)
            loss = loss_fn(pred.double(), y_vals.double())
            loss_lst.append(loss.item())
            # Backpropagation
            optimizer.zero_grad() # zero the gradients
            loss.backward() # backpass
            optimizer.step() # step
            n+=1
            if n > num_iters:
                break
            if batch % 100 == 0:
                print("")
                unique, counts = np.unique(y_vals,return_counts = True)
                d = dict(zip(unique,counts))
                print("Unique Counts Dict:", d)
                loss_avg = np.average(loss_lst[-100:])
                loss, current = loss.item(), batch * X_local.shape[0]
                print(f"loss: {loss_avg:>7f}  [{current:>5d}/{size:>5d}]")
    return loss_lst