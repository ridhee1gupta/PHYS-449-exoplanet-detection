from networks import FullyConnectedNetwork, train_net
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from create_datasets import dataset_maker
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report

"""
Using the results of the gridsearch this script will use those parameters and create .npy files of the testing and training dataset
"""

# Optimal Grid Search Parameters
lr = 0.001
batch_size = 256
oversample_rate = 2

# Load train data
train_local = 'NpyData/TrainData/local_view.npy'
train_global = 'NpyData/TrainData/global_view.npy'
train_impact = 'NpyData/TrainData/tce_impact.npy'
train_depth = 'NpyData/TrainData/tce_depth.npy'
train_labels = 'NpyData/TrainData/av_training_set.npy'
n_oversample = int(oversample_rate * 26)
train_dataset, train_dataloader = dataset_maker(train_local,train_global,train_impact,train_depth, train_labels, batch_size, n_oversample)

fc_network = FullyConnectedNetwork()
fc_network.double()
loss = nn.BCELoss()
optimizer = torch.optim.Adam(fc_network.parameters(), lr=lr)
loss_lst = train_net(train_dataloader, fc_network, loss, optimizer, num_iters = 4000)

# Load Test Data
test_local = 'NpyData/TestData/Test-local_view.npy'
test_global = 'NpyData/TestData/Test-global_view.npy'
test_impact = 'NpyData/TestData/Test-tce_impact.npy'
test_depth = 'NpyData/TestData/Test-tce_depth.npy'
test_labels = 'NpyData/TestData/Test-av_training_set.npy'
test_dataset, test_dataloader = dataset_maker(test_local, test_global, test_impact, test_depth, test_labels, batch_size=2755, n_oversample = 0)
for batch, (test_X_local, test_X_global, test_impact, test_depth, testy) in enumerate(test_dataloader):
    pred = fc_network(test_X_local, test_X_global, test_impact, test_depth)
    pred_arr = pred.detach().numpy().reshape(pred.shape[0])
    testy_npy = testy.detach().numpy().reshape(testy.shape[0])
    print(classification_report(testy_npy, np.round(pred_arr)))
    precisions = []
    recalls = []
    fscores = []
    for thresh in np.linspace(0, 1, 1000):
        precision, recall, fscore, support = precision_recall_fscore_support(testy_npy, (pred_arr > thresh),
                                                                             zero_division=1)
        precisions.append(precision[1])
        recalls.append(recall[1])
        fscores.append(fscore[1])
    fig, ax = plt.subplots(1, 2, figsize=(19.2, 10.8))
    max_fscore_idx = np.where(fscores == np.max(fscores))[0][0]
    max_fscore = np.round(np.max(fscores), 2)
    fig.suptitle("Precision vs Recall and Loss Plots for Testing Data")
    ax[0].plot(recalls, precisions, color='black')
    ax[0].scatter(recalls[max_fscore_idx], precisions[max_fscore_idx], s=80, edgecolors='r',
                  facecolors='none')

    ax[0].annotate('F1 Score = %s' % max_fscore,
                   (recalls[max_fscore_idx] + .01, precisions[max_fscore_idx] + .01))
    ax[1].plot(pd.Series(loss_lst).rolling(25).mean().values, color='black')

    fig.savefig('outputs/test_results.png')
    plt.close()


output_arr = np.c_[pred_arr,testy_npy]

np.save("predictions.npy", output_arr)
x = 1
