#################################################### Purpose ####################################################
"""
This script runs a grid search to find optimal hyperparameters
"""
#################################################### Imports ####################################################
from networks import *
import numpy as np
import gc
import torch
from create_datasets import dataset_maker
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

#################################################### Code ####################################################
# Load Validation Data
val_local = 'NpyData/ValData/Val-local_view.npy'
val_global = 'NpyData/ValData/Val-global_view.npy'
val_impact = 'NpyData/ValData/Val-tce_impact.npy'
val_depth = 'NpyData/ValData/Val-tce_depth.npy'
val_labels = 'NpyData/ValData/Val-av_training_set.npy'

val_dataset, val_dataloader = dataset_maker(val_local, val_global, val_impact, val_depth, val_labels, batch_size=2774, n_oversample = 0)
val_X_local = val_dataset.X_local
val_X_global = val_dataset.X_global
val_impact = val_dataset.impact
val_depth = val_dataset.depth

for batch, (X_local, X_global, impact, rad_rat, y_vals) in enumerate(val_dataloader):
    x = 1



learning_rates = [1e-3, 1e-4, 1e-5]
batch_sizes = [16,32,64,128,256]
oversample_rates = [0, 0.5,1,2]
b = []
l = []
o = []
f1 = []
for lr in learning_rates:
    for batch_size in batch_sizes:
        for os_rate in oversample_rates:
            b.append(batch_size)
            l.append(lr)
            o.append(os_rate)
            print("Starting Batch Size %s,  Oversampling Rate %s and learning rate %s" % (batch_size, os_rate, lr))
            # Load data
            train_local = 'NpyData/TrainData/local_view.npy'
            train_global = 'NpyData/TrainData/global_view.npy'
            train_impact = 'NpyData/TrainData/tce_impact.npy'
            train_depth = 'NpyData/TrainData/tce_depth.npy'
            train_labels = 'NpyData/TrainData/av_training_set.npy'
            n_oversample = int(os_rate * 26)
            train_dataset, train_dataloader = dataset_maker(train_local,train_global,train_impact,train_depth, train_labels, batch_size, n_oversample)

            fc_network = FullyConnectedNetwork()
            fc_network.double()
            loss = nn.BCELoss()
            optimizer = torch.optim.Adam(fc_network.parameters(), lr=lr)
            loss_lst = train_net(train_dataloader, fc_network, loss, optimizer, num_iters = 1000)

            # Compute prediction and loss
            for batch, (val_X_local, val_X_global, val_impact, val_depth, valy) in enumerate(val_dataloader):
                pred = fc_network(val_X_local, val_X_global, val_impact, val_depth)
                pred_arr = pred.detach().numpy().reshape(pred.shape[0])
                valy_npy = valy.detach().numpy().reshape(valy.shape[0])
                print(classification_report(valy_npy, np.round(pred_arr)))
                precisions = []
                recalls = []
                fscores = []
                for thresh in np.linspace(0, 1, 1000):
                    precision, recall, fscore, support = precision_recall_fscore_support(valy_npy, (pred_arr > thresh),
                                                                                         zero_division=1)
                    precisions.append(precision[1])
                    recalls.append(recall[1])
                    fscores.append(fscore[1])
                fig, ax = plt.subplots(1, 2, figsize=(19.2, 10.8))
                max_fscore_idx = np.where(fscores == np.max(fscores))[0][0]
                max_fscore = np.round(np.max(fscores), 2)
                f1.append(max_fscore)
                fig.suptitle("Batch size = %s, Learning Rate = %s, Oversampling Rate = %s \n Best F1Score = %s" % (
                batch_size, lr, os_rate, max_fscore))
                ax[0].plot(recalls, precisions, color='black')
                ax[0].scatter(recalls[max_fscore_idx], precisions[max_fscore_idx], s=80, edgecolors='r',
                              facecolors='none')

                ax[0].annotate('F1 Score = %s' % max_fscore,
                               (recalls[max_fscore_idx] + .01, precisions[max_fscore_idx] + .01))
                ax[1].plot(pd.Series(loss_lst).rolling(25).mean().values, color='black')

                fig.savefig('outputs/gridsearch_plots/batch%s_lr%s_osrate%s.png' % (batch_size, lr, os_rate))
                plt.clf()
                plt.cla()
                plt.close('all')
                plt.close(fig)
                gc.collect()
df = pd.DataFrame({'batch size': b, 'learning rate':l,'Oversample rate':o,'max f1 score': f1})

df.to_csv('outputs/gridsearch_results.csv', index = False)