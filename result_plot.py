import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

predictions = np.load('predictions.npy')
print(predictions)
#print(predictions[0][0])
pred = predictions[:,0]
labels = predictions[:,1]


"""
hist, bin_edges = np.histogram(predictions[:,0])
print(hist)
print(bin_edges)
print(hist.size, bin_edges.size)
"""
n, bins, patches = plt.hist(pred, bins=20, log=True)
print(n, bins)
labels = labels.tolist()
for i in range(len(bins)):
    label_i = []
    for j in range(len(pred)):
        print(pred[j], bins[i], bins[i-1])
        labels_j = []
        if pred[j] <= bins[i] and pred[j] > bins[i-1]:
            in_bin = True
            labels = labels
            print(labels)
        else:
            print(j)
            labels.pop(j)
            print(labels)

        #print(label_i)


#print(patches)
cm = plt.cm.get_cmap('plasma')

for n, p in enumerate(patches):
    

"""
fig, ax = plt.subplots(1, 2, figsize=(40, 40))

ax[0].hist(predictions[:,0], bins=20, log=True)
ax[0].set_xlabel('Prediction')
ax[0].set_ylabel('Number of TCEs')
ax[0].set_title('Test data predictions')
plt.colorbar(ax.images[0], ax=ax, label='Fraction of TCEs in Each Bin Labeled Candidates')

plt.show()
"""
