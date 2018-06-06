from utility import loadPickle, writePickle
from svm import getTrainDataAndLabel
from bow import encodeBatch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch
import copy


class ANN(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(ANN, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, X):
        h_relu = self.linear1(X).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


if __name__ == '__main__':
    # get bag of words
    bow = loadPickle('./bag-of-words/bow_500').cluster_centers_

    # define #-dimensions
    N, D_in, H, D_out = 50, 500, 1000, 10

    trn_encoded, labels = getTrainDataAndLabel(bow)

    # Split data to train set and dev set
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(trn_encoded, labels):
        trn_data = trn_encoded[train_index]
        trn_labels = labels[train_index]

        dev_data = trn_encoded[test_index]
        dev_labels = labels[test_index]

    # print trn_data.shape, dev_data.shape

    tst = loadPickle('./cifar-10-batches-py/test_batch')
    tst_encoded = encodeBatch(tst, bow)

    X = torch.from_numpy(trn_data).float()
    Y = torch.from_numpy(trn_labels).long()
    X_dev = torch.from_numpy(dev_data).float()
    Y_dev = torch.from_numpy(dev_labels).long()
    X_test = torch.from_numpy(np.array(tst_encoded)).float()
    Y_test = torch.from_numpy(np.array(tst['labels'])).long()

    model = ANN(D_in, H, D_out)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    acc_go_down = 0
    max_dev_acc = float('-inf')
    bestModel = None
    for t in range(500):
        for iteration in range(X.shape[0] // N):
            start = iteration * N
            end = start + N
            x = X[start:end, :]
            y_pred = model(x)

            y = Y[start:end]
            loss = criterion(y_pred, y)
            # print t, loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Y_dev_pred = model(X_dev)
        dev_acc = 100 * \
            accuracy_score(Y_dev_pred.data.numpy().argmax(
                axis=1), Y_dev.numpy())

        if dev_acc > max_dev_acc:
            acc_go_down = 0
            bestModel = copy.deepcopy(model)
        else:
            acc_go_down += 1
            if acc_go_down == 20:
                break

        Y_test_pred = model(X_test)
        print t, 'test accuracy', 100 * \
            accuracy_score(Y_test_pred.data.numpy().argmax(
                axis=1), Y_test.numpy())

    writePickle(bestModel, './models/ann/ann_3layer_bow_500')
