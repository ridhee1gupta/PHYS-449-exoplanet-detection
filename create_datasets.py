#################################################### Purpose ####################################################
"""
This script is used to create the datasets that will be used to sort data and input batches into the model
"""
#################################################### Imports ####################################################
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

#################################################### Code ####################################################
def dataset_maker(local_path,global_path,impact_path,depth_path, labels_path, batch_size, n_oversample):
    # Load data
    test_local_views = np.load(local_path)
    test_global_views = np.load(global_path)
    test_impact = np.load(impact_path)
    test_depth = np.load(depth_path)
    test_labels = np.load(labels_path)[:,1]


    test_labels = (test_labels == 67)
    test_labels = test_labels.reshape(test_labels.shape[0], 1)

    test_local_views_l = test_local_views.tolist()
    test_global_views_l = test_global_views.tolist()
    test_impact_l = test_impact.tolist()
    test_depth_l = test_depth.tolist()
    test_labels_l = test_labels.tolist()

    pos_idxs = np.where(test_labels[:, 0])[0]
    for idx in pos_idxs:
        for n in range(n_oversample):
            test_local_views_l.append(test_local_views_l[idx])
            test_global_views_l.append(test_global_views_l[idx])
            test_impact_l.append(test_impact_l[idx])
            test_depth_l.append(test_depth_l[idx])
            test_labels_l.append(test_labels_l[idx])
            # local_views = np.insert(local_views, -1, local_views[idx], axis = 0)
            # global_views = np.insert (global_views,-1,global_views[idx], axis = 0)
            # impact = np.insert(impact, -1,impact[idx], axis = 0)
            # depth = np.insert(depth, -1, depth[idx],axis = 0)
            # labels = np.insert(labels, -1,  labels[idx],axis = 0)

    test_impact = np.array(test_impact_l)
    test_depth = np.array(test_depth_l)
    test_local_views = np.array(test_local_views_l)
    test_local_views = torch.DoubleTensor(test_local_views.reshape(test_local_views.shape[0], 1, test_local_views.shape[1]))
    test_global_views = np.array(test_global_views_l)
    test_global_views = torch.DoubleTensor(test_global_views.reshape(test_global_views.shape[0], 1, test_global_views.shape[1]))

    test_labels = np.array(test_labels_l)

    test_y = torch.FloatTensor(test_labels)

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
            return local_vals, global_vals, impact_vals, depth_vals, labels


    data = Data(test_local_views, test_global_views, test_impact, test_depth, test_y)

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return data, dataloader