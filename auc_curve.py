import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

pred_arr = np.load('predictions.npy')[:,0]
testy_npy = np.load('predictions.npy')[:,1]


fig, ax = plt.subplots(2, tight_layout = True, figsize = (8,12))
PrecisionRecallDisplay.from_predictions(testy_npy,pred_arr, ax = ax[0])

ax[0].set_ylabel('Precision (Reliability)')
ax[0].set_xlabel('Recall (Completeness)')

RocCurveDisplay.from_predictions(testy_npy,pred_arr, ax = ax[1])
ax[1].set_ylabel('Recall (Completeness)')
ax[1].set_xlabel('False Positive Rate')

plt.savefig('outputs/RecallvsPrecision_and_AUC.png')

