import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

predictions = np.load('predictions.npy')
print(predictions)
pred = predictions[:,0]
labels = predictions[:,1]

fig = plt.figure(figsize = (25.6,14.4))
gs = gridspec.GridSpec(2,3, height_ratios = [2, 1], width_ratios = [1,1,0.2])

ax1 = fig.add_subplot(gs[0,:2])
n, bins, patches = ax1.hist(pred, bins=20, log=True, edgecolor = 'black', linewidth = 1.2)
bin_starts = np.linspace(0,0.95,20)
cm = plt.cm.get_cmap('plasma')
for n, bin_start in enumerate(bin_starts):
    bin_end = bin_start + 0.05
    bin_idxs, = np.where((pred >= bin_start) & (pred < bin_end))
    pct_exo = labels[bin_idxs].sum()/len(bin_idxs)
    plt.setp(patches[n], 'facecolor', cm(pct_exo))

ax2 = fig.add_subplot(gs[1,0])
n, bins, patches = ax2.hist(np.log(pred), bins = 37, range = (-100, 0), edgecolor = 'black', linewidth = 1.2)
bin_starts = np.linspace(-100,-100/37,37)
ax2 = plt.gca()
for n, bin_start in enumerate(bin_starts):
    bin_end = bin_start + 100/37
    bin_idxs, = np.where((np.log(pred) >= bin_start) & (np.log(pred) < bin_end))
    pct_exo = labels[bin_idxs].sum()/len(bin_idxs)
    plt.setp(patches[n], 'facecolor', cm(pct_exo))


ax3 = fig.add_subplot(gs[1,1])
n, bins, patches = ax3.hist(np.log(pred), bins = 37, range = (-3, 0), edgecolor = 'black', linewidth = 1.2)
bin_starts = np.linspace(-3,-3/37,37)
for n, bin_start in enumerate(bin_starts):
    bin_end = bin_start + 3/37
    bin_idxs, = np.where((np.log(pred) >= bin_start) & (np.log(pred) < bin_end))
    pct_exo = labels[bin_idxs].sum()/len(bin_idxs)
    plt.setp(patches[n], 'facecolor', cm(pct_exo))


axclrbar = plt.subplot(gs[:,2])
from matplotlib.colorbar import Colorbar
from matplotlib.cm import ScalarMappable
sm = ScalarMappable(cmap=cm, norm = plt.Normalize(0,1))
cb = Colorbar(ax = axclrbar, mappable = sm)

ax1.set_xlabel('Prediction', fontsize = 16)
ax1.set_ylabel('Number of TCEs', fontsize = 16)
ax2.set_xlabel('Log Prediction', fontsize = 16)
ax2.set_ylabel('Number of TCEs', fontsize = 16)
ax3.set_xlabel('Log Prediction', fontsize = 16)
ax3.set_ylabel('Number of TCEs', fontsize = 16)
axclrbar.set_ylabel('Fraction of TCEs in Each Bin\nLabeled Candidates', fontsize = 16)

plt.savefig('outputs/Histogram.png')