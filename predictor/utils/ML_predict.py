import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

csv_col = ['path', 'left', 'middle', 'right', 'class']


def get_data():
    # Read dataset to pandas dataframe
    dataset = pd.read_csv('training.csv', names=csv_col)
    X_train = dataset.iloc[1:, 1:-1].values
    #X_train = [[x.strip('][').split(", ") for x in values] for values in X_train]
    y_train = dataset.iloc[1:, 4].values

    dataset = pd.read_csv('test.csv', names=csv_col)
    X_test = dataset.iloc[1:, 1:-1].values
    #X_test = [[x.strip('][').split(", ") for x in values] for values in X_test]
    y_test = dataset.iloc[1:, 4].values

    # Scale features
    #X_train = [[[float(value) / 255.0 for value in y] for y in values] for values in X_train]
    #X_test = [[[float(value) / 255.0 for value in y] for y in values] for values in X_test]

    #X_train = [np.asarray(values).flatten() for values in X_train]
    #X_test = [np.asarray(values).flatten() for values in X_test]

    X_train = [[float(value) / 255.0 for value in values] for values in X_train]
    X_test = [[float(value) / 255.0 for value in values] for values in X_test]

    return X_train, y_train, X_test, y_test


def train_knn():
    # Read dataset
    X_train, y_train, X_test, y_test = get_data()

    error = []
    # Calculating error for K values between 1 and 40
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))


    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()

    best_k = np.argmin(error)

    # Train
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)

    # Evaluate
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save
    with open('knn_model', 'wb') as model:
        pickle.dump(classifier, model)


def test_knn():
    # Read dataset
    _, _, X_test, y_test = get_data()

    # Load
    with open('knn_model', 'rb') as model:
        classifier = pickle.load(model)

    # Evaluate
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def test_images():

    # Read dataset to pandas dataframe
    dataset = pd.read_csv('test.csv', names=csv_col)
    X_test = dataset.iloc[1:, 0:-1].values
    y_test = dataset.iloc[1:, 4].values

    # Load
    with open('knn_model', 'rb') as model:
        classifier = pickle.load(model)
    
    preds = []
    count = 0
    pos = 0
    for idx, feature in enumerate(X_test):
        # For PCA with more than one value per left,middle,right
        if False:
            X_test = [values.strip('][').split(", ") for values in feature[1:]]
            feat = [[float(value)/ 255.0 for value in values] for values in X_test]
            feat = np.asarray(feat).flatten()
        # For mean value with simple scale
        if True:
            feat = [float(value)/ 255.0 for value in feature[1:]]
        # For mean value with given scale and offset
        if False:
            feat = []
            for idx, value in enumerate(feature[1:]):
                val = (float(value) - u_x[idx]) / s_x[idx]
                feat.append(val)

        if classifier.predict([feat]) == y_test[idx]:
            pos += 1
        #elif y_test[idx] == 'Dset-1.0':
        #    img = feature[0]
        #    print(img)
        count += 1
    print("ACCURACY:", pos/count)
        
    
def main():
    parser = argparse.ArgumentParser(description='Label dataset')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    train = args.train
    test = args.test

    if train:
        train_knn()
    if test:
        test_images()


if __name__ == "__main__":
    main()
