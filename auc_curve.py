from networks import *
import numpy as np
import gc
from torch.utils.data import DataLoader, Dataset
import torch
from create_datasets import dataset_maker
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

test_local = 'NpyData/TestData/Test-local_view.npy'
test_global = 'NpyData/TestData/Test-global_view.npy'
test_impact = 'NpyData/TestData/Test-tce_impact.npy'
test_depth = 'NpyData/TestData/Test-tce_depth.npy'
test_labels = 'NpyData/TestData/Test-av_training_set.npy'
test_dataset, test_dataloader = dataset_maker(test_local, test_global, test_impact, test_depth, test_labels, batch_size=2755, n_oversample = 0)
for batch, (test_X_local, test_X_global, test_impact, test_depth, testy) in enumerate(test_dataloader):
    gc.collect()
    #pred = fc_network(test_X_local, test_X_global, test_impact, test_depth)
    pred_arr = np.load('predictions.npy')[0]
    testy_npy = np.load('predictions.npy')[1]
    #print(classification_report(testy_npy, np.round(pred_arr)))
    tn = []
    fp = []
    fn = []
    tp = []
    for thresh in np.linspace(0, 1, 1000):
        tn, fp, fn, tp = confusion_matrix(testy_npy, (pred_arr > thresh)).ravel()
        print(tn, fp, fn, tp)
        #tn.append(tn[1])
        #print(tn)
        #fp.append(fp[1])
        #fn.append(fn[1])
        #tp.append(tp[1])
    """
    plt.plot(fscores, recalls, color='black')
    plt.xlabel("False Positive Rate")
    plt.ylabel("Recalls")
    plt.title("ROC Curve")
    plt.savefig('outputs/roc_curve.png')
    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(19.2, 10.8))
    max_fscore_idx = np.where(fscores == np.max(fscores))[0][0]
    max_fscore = np.round(np.max(fscores), 2)
    fig.suptitle("Precision vs Recall and Loss Plots for Testing Data")
    ax[0].plot(recalls, precisions, color='black')
    ax[0].scatter(recalls[max_fscore_idx], precisions[max_fscore_idx], s=80, edgecolors='r',
                  facecolors='none')

    ax[0].annotate('F1 Score = %s' % max_fscore,
                   (recalls[max_fscore_idx] + .01, precisions[max_fscore_idx] + .01))
    ax[0].set_xlabel("Precision")
    ax[0].set_ylabel("Recall")
    ax[1].plot(pd.Series(loss_lst).rolling(25).mean().values, color='black')
    ax[1].set_xlabel("Number of epochs")
    ax[1].set_ylabel("Loss")

    fig.savefig('outputs/test_results.png')
    plt.close()


output_arr = np.c_[pred_arr,testy_npy]

np.save("predictions.npy", output_arr)
x = 1
"""
