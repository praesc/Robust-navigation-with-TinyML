import os
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA

PLOT = 0
# Possibilities: left, middle, right
PLOT_PART = 'left'

categories = ['Crossing',  'EndSpeedLimit',  'FinishLine',  'LeftTurn',  'RightTurn',  'StartSpeedLimit',  'Straight']
csv_col = ['path', 'left', 'middle', 'right', 'class']
frames = dict()
fname = 'test'


def save_to_csv(frames: dict, fname: str):
    paths = []
    lefts = []
    middles = []
    rights = []
    classes = []
    # Shaffle
    l = list(frames.items())
    random.shuffle(l)
    frames = dict(l)

    for key, frame in frames.items():
        paths.append(frame[csv_col[0]])
        lefts.append(frame[csv_col[1]])
        middles.append(frame[csv_col[2]])
        rights.append(frame[csv_col[3]])
        classes.append(frame[csv_col[4]])

    #lefts = PrincipalComponentAnalysis(lefts).tolist()
    #middles = PrincipalComponentAnalysis(middles).tolist()
    #rights = PrincipalComponentAnalysis(rights).tolist()

    df = pd.DataFrame(dict([(csv_col[0], paths),
                    (csv_col[1], lefts),
                    (csv_col[2], middles),
                    (csv_col[3], rights),
                    (csv_col[4], classes)]))
    df.to_csv(fname)


def PrincipalComponentAnalysis(X, pca = 3):
    if pca == None:
        pca_components = np.identity(X.shape[1])
    else:
        pca = PCA(n_components=pca)
        X = pca.fit_transform(X)
        pca_components = pca.components_ # These are needed at runtime
    print(pca.explained_variance_ratio_)

    return X

#########################################################################################
if PLOT:
    fig, axs = plt.subplots(len(categories))

lefts = []
middles = []
rights = []
for idx, category in enumerate(categories) :
    data1 = []
    class_ = 'Dset-1.0'
    for file in os.listdir(os.path.join('../datasets/Dset-1.0', fname, category)):
        f_path = os.path.join('../datasets/Dset-1.0', fname, category, file)
        img = cv2.imread(f_path, 0)
        left = np.mean(np.asarray(img[0, 0:47]))
        middle = np.mean(np.asarray(img[0, 47:96]))
        right = np.mean(np.asarray(img[0, 96:]))

        lefts.append(left)
        middles.append(middle)
        rights.append(right)

        if PLOT:
            data1.append(PLOT_PART)

        frames[f_path[12:]] = {csv_col[0]: f_path[12:],
                          csv_col[1]: left,
                          csv_col[2]: middle,
                          csv_col[3]: right,
                          csv_col[4]: class_}

    data2 = []
    class_ = 'Dset-2.0'
    for file in os.listdir(os.path.join('../datasets/Dset-2.0', fname, category)):
        f_path = os.path.join('../datasets/Dset-2.0', fname, category, file)
        img = cv2.imread(f_path, 0)
        left = np.mean(np.asarray(img[0, 0:47]))
        middle = np.mean(np.asarray(img[0, 47:96]))
        right = np.mean(np.asarray(img[0, 96:]))

        lefts.append(left)
        middles.append(middle)
        rights.append(right)

        if PLOT:
            data2.append(PLOT_PART)

        frames[f_path[12:]] = {csv_col[0]: f_path[12:],
                          csv_col[1]: left,
                          csv_col[2]: middle,
                          csv_col[3]: right,
                          csv_col[4]: class_}


    if PLOT:
        axs[idx].plot(data1, 'r--',data2 , 'bs')
        axs[idx].set_title(category, fontsize=10)

save_to_csv(frames, fname +'-try'+ '.csv')

if PLOT:
    plt.show()
