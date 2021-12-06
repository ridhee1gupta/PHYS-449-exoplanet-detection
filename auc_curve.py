from networks import *
import numpy as np
import gc
from torch.utils.data import DataLoader, Dataset
import torch
from create_datasets import dataset_maker
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

pred_arr = np.load('predictions.npy')[:,0]
testy_npy = np.load('predictions.npy')[:,1]
#print(classification_report(testy_npy, np.round(pred_arr)))
tn = []
fp = []
fn = []
tp = []
fp_rates = []
recalls = []
precisions = []

for thresh in np.linspace(0, 1, 1000):
    tn, fp, fn, tp = confusion_matrix(testy_npy, (pred_arr > thresh)).ravel()
    fp_rates.append(fp/(fp+tn))
    recalls.append(tp/(tp+fn))
    precisions.append(tp/(tp+fp))
# for thresh in np.logspace(-1000, 0, 1000):
#     tn, fp, fn, tp = confusion_matrix(testy_npy, (pred_arr > thresh)).ravel()
#     fp_rates.append(fp/(fp+tn))
#     recalls.append(tp/(tp+fn))
#     precisions.append(tp/(tp+fp))

# fp_rate_inds = np.array(fp_rates).argsort()
# recalls = np.array(recalls)[fp_rate_inds[::-1]]
# fp_rates = np.array(fp_rates)[fp_rate_inds[::-1]]
# precisions = np.array(precisions)[fp_rate_inds[::-1]]

fig, ax = plt.subplots(2, tight_layout = True, figsize = (8,12))
ax[0].plot(recalls,precisions, color = 'black')
ax[0].set_ylabel('Precision (Reliability)')
ax[0].set_xlabel('Recall (Completeness)')


precision_inds = np.array(precisions).argsort()
recalls = np.array(recalls)[precision_inds[::-1]]
fp_rates = np.array(fp_rates)[precision_inds[::-1]]
precisions = np.array(precisions)[precision_inds[::-1]]
ax[1].plot(fp_rates,recalls, color = 'black')
ax[1].set_ylabel('Recall (Completeness)')
ax[1].set_xlabel('False Positive Rate')

plt.savefig('outputs/RecallvsPrecision_and_AUC.png')
x = 1
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
