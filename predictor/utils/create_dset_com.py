import os
import cv2
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA

categories = ['Crossing',  'EndSpeedLimit',  'FinishLine',  'LeftTurn',  'RightTurn',  'StartSpeedLimit',  'Straight']
csv_col = ['path', 'data', 'class']
frames = dict()
fname = 'test'


def save_to_csv(frames: dict, fname: str):
    paths = []
    data = []
    for idx in range(143):
        data.append([])
    classes = []
    # Shaffle
    l = list(frames.items())
    random.shuffle(l)
    frames = dict(l)

    for key, frame in frames.items():
        paths.append(frame[csv_col[0]])
        values = frame[csv_col[1]][0]
        for idx, element in enumerate(values):
            data[idx].append(element)
        classes.append(frame[csv_col[2]])

    fram = []
    fram.append((csv_col[0], paths))
    for idx, element in enumerate(data):
        fram.append((idx, element))
    fram.append((csv_col[2], classes))

    df = pd.DataFrame(dict(fram))
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
'''for idx, category in enumerate(categories) :
    class_ = 'Dset-1.0'
    for file in os.listdir(os.path.join('../datasets/Dset-1.0', fname, category)):
        f_path = os.path.join('../datasets/Dset-1.0', fname, category, file)
        img = cv2.imread(f_path, 0)
        data = np.asarray(img)

        frames[f_path[12:]] = {csv_col[0]: f_path[12:],
                          csv_col[1]: data,
                          csv_col[2]: class_}

    class_ = 'Dset-2.0'
    for file in os.listdir(os.path.join('../datasets/Dset-2.0', fname, category)):
        f_path = os.path.join('../datasets/Dset-2.0', fname, category, file)
        img = cv2.imread(f_path, 0)
        data = np.asarray(img)

        frames[f_path[12:]] = {csv_col[0]: f_path[12:],
                          csv_col[1]: data,
                          csv_col[2]: class_}

    class_ = 'Dset-1.0'
    for file in os.listdir(os.path.join('../datasets/Dset-1.5', fname, category)):
        f_path = os.path.join('../datasets/Dset-1.5', fname, category, file)
        img = cv2.imread(f_path, 0)
        data = np.asarray(img)

        frames[f_path[12:]] = {csv_col[0]: f_path[12:],
                          csv_col[1]: data,
                          csv_col[2]: class_}
'''
frames = dict()
for idx, category in enumerate(['right',  'wrong']) :
    class_ = category
    for file in os.listdir(os.path.join('../datasets/architecture', 'test', category)):
        f_path = os.path.join('../datasets/architecture', 'test', category, file)
        img = cv2.imread(f_path, 0)
        data = np.asarray(img)

        frames[f_path[12:]] = {csv_col[0]: f_path[12:],
                          csv_col[1]: data,
                          csv_col[2]: class_}

save_to_csv(frames, 'test' +'-complete-arch-dsets'+ '.csv')

frames = dict()
for idx, category in enumerate(['right',  'wrong']) :
    class_ = category
    for file in os.listdir(os.path.join('../datasets/architecture', 'training', category)):
        f_path = os.path.join('../datasets/architecture', 'training', category, file)
        img = cv2.imread(f_path, 0)
        data = np.asarray(img)

        frames[f_path[12:]] = {csv_col[0]: f_path[12:],
                          csv_col[1]: data,
                          csv_col[2]: class_}

save_to_csv(frames, 'training' +'-complete-arch-dsets'+ '.csv')