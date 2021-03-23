from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from sklearn import metrics
import matplotlib.cm as cm
from time import time
import pandas as pd
import numpy as np


def oversampling(x, c):

    """

    :param x: training dataset in pandas format.
    :param c: a string with the name of the output feature.
    :return: the oversampled training dataset.
    """

    liste = list(x[c].unique())
    over = x[c].value_counts(normalize=False).index[0]
    label_0 = x[x[c] == over]
    liste.remove(int(over))

    taille = label_0.shape[0]
    new_0 = label_0.sample(n=taille, replace=False)

    for i in liste:
        label_1_1 = x[x[c] == i]
        new_1_1 = label_1_1
        new_1_2 = label_1_1.sample(n=(taille - label_1_1.shape[0]), replace=True)
        new_1 = pd.concat([new_1_1, new_1_2], ignore_index=True)
        new_0 = pd.concat([new_0, new_1], ignore_index=True)

    return new_0.sample(n=new_0.shape[0], replace=False)


Dataset = pd.read_csv('./data/experimental_data.csv', sep=";")

# Chose te desired output for heterogeneity learning
# column = 'mu' or 'mg'
column = 'mu'
Dataset = Dataset[['Active Material [%]', 'Solid Content [%]', 'Comma Gap [μm]', column]]

# Split the dataset and standardize the data.
Train, Test = train_test_split(Dataset, train_size=0.8)
Train = oversampling(Train, column)
XTrain = Train.drop(columns=column)
XTest = Test.drop(columns=column)
YTrain = Train[column]
YTest = Test[column]

scaler = StandardScaler()
scaler.fit(Dataset.drop(columns=[column]))
XTrain = pd.DataFrame(scaler.transform(XTrain), columns=XTrain.columns)
XTest = pd.DataFrame(scaler.transform(XTest), columns=XTest.columns)

# Training step
print("_" * 60)
print("Training")
t0 = time()
method = GaussianNB()
method.fit(XTrain, YTrain.values.ravel())
print("Train score : %0.3f" % method.score(XTrain, YTrain))
train_time = time() - t0
print("Train time : %0.3fs" % train_time)

# Testing step
print("\nTesting")
predictions = method.predict(XTest)
score = metrics.f1_score(YTest.values.ravel(), predictions, average='micro')
print("F1 score : %0.3f" % score)
fpr, tpr, _ = metrics.roc_curve(YTest.values.ravel(), method.predict_proba(XTest)[:, 1])
roc_auc = metrics.auc(fpr, tpr)
print("AUC : %0.3fs" % roc_auc)
print("\nConfusion matrix : ")
print(confusion_matrix(YTest.values.ravel(), predictions))
print("\n\n")


if column == 'mu':
    title = 'Thickness'
else:
    title = 'Loading'

fig = plt.figure(figsize=(15, 10))
fig.subplots_adjust(hspace=.4, wspace=.3)
plt.suptitle('Influence of manufacturing parameters on electrode heterogeneity in '+title, fontsize=22, fontweight="bold")
gs1 = gridspec.GridSpec(2, 2)
axs = []
titles = ['Active Material [%] : 92.7', 'Active Material [%] : 94',
          'Active Material [%] : 95', 'Active Material [%] : 96']
border1 = []
border2 = []

for am, num in zip([92.7, 94, 95, 96], range(1, 5)):

    axs.append(fig.add_subplot(gs1[num - 1]))

    xx, yy = np.meshgrid(np.linspace(50, 400, 100),
                         np.linspace(0.55, 0.72, 100))

    xpred = pd.DataFrame(np.array([np.repeat(am, xx.ravel().size) for _ in range(1)] + [yy.ravel(), xx.ravel()]).T,
                         columns=["0", "1", "2"])
    xpred = np.array(xpred)

    xpred = scaler.transform(xpred)

    pred = method.predict_proba(xpred)[:, 1].reshape(xx.shape)

    pred2 = method.predict(xpred).reshape(xx.shape)

    border1.append(pd.DataFrame(pred).max().max())
    border2.append(pd.DataFrame(pred).min().min())

    for m in range(100):
        for n in range(100):

            if pred2[m][n] == 'Homogeneous':
                pred2[m][n] = 1
            if pred2[m][n] == 'Heterogeneous':
                pred2[m][n] = 2

    cmap = plt.cm.seismic
    bounds = np.linspace(0, 1, 1500)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    axs[-1].imshow(pred,
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
                   origin='lower', cmap=cmap, norm=norm)

    if num == 1 or num == 3:
        plt.ylabel('Solid Content [%]', fontsize=17, labelpad=25)
    if num == 3 or num == 4:
        plt.xlabel('Comma Gap [μm]', fontsize=17, labelpad=25)
    plt.title(titles[num - 1], fontsize=18, color='k', fontweight="bold")
    plt.axis([50, 400, 0.55, 0.725])

co = cm.ScalarMappable(cmap=plt.cm.seismic_r)
co.set_array([min(border2), max(border1)])
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.7])
cbar = fig.colorbar(co, cax=cbar_ax, ticks=[min(border2), (max(border1) + min(border2)) / 2, max(border1)])
cbar.ax.set_ylabel('Probability of heterogeneous electrode', rotation=270, fontsize=19, labelpad=25)
cbar.ax.set_yticklabels(["0", '0.5', "1"])
cbar.ax.tick_params(labelsize=17)
plt.show()