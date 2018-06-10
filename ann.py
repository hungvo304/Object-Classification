from utility import loadPickle, writePickle, stratifiedSplitTrainSet
from svm import getTrainDataAndLabel
from sklearn.metrics import accuracy_score
from bow import encodeImage
import torch
import copy
import cv2
import numpy as np


class ANN(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(ANN, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, X):
        h_relu = self.linear1(X).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


class NeuralNet(object):
    def __init__(self, N, D_in, H, D_out):
        self.N = N
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.model = ANN(D_in, H, D_out)

    def fit(self, trn_encoded, labels):
        # Split data to train set and dev set
        trn_data, trn_labels, dev_data, dev_labels = stratifiedSplitTrainSet(
            trn_encoded, labels, num_splits=1)

        # Convert numpy array to torch tensor
        X = torch.from_numpy(trn_data).float()
        Y = torch.from_numpy(trn_labels).long()
        X_dev = torch.from_numpy(dev_data).float()
        Y_dev = torch.from_numpy(dev_labels).long()

        # Train model
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=1e-4, momentum=0.9)
        acc_go_down = 0
        max_dev_acc = float('-inf')
        bestModel = None
        for t in range(500):
            for iteration in range(X.shape[0] // self.N):
                start = iteration * self.N
                end = start + self.N
                x = X[start:end, :]
                y_pred = self.model(x)

                y = Y[start:end]
                loss = criterion(y_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            Y_dev_pred = self.model(X_dev)
            dev_acc = 100 * \
                accuracy_score(Y_dev_pred.data.numpy().argmax(
                    axis=1), Y_dev.numpy())
            print t, 'dev accuracy', dev_acc

            if dev_acc > max_dev_acc:
                acc_go_down = 0
                bestModel = copy.deepcopy(self.model)
            else:
                acc_go_down += 1
                if acc_go_down == 20:
                    break

        self.model = copy.deepcopy(bestModel)

    def predict(self, X):
        result_classes = []
        for row in X:
            x_test = torch.from_numpy(row).float()
            result_classes.append(torch.argmax(self.model(x_test)))

        return np.array(result_classes)

    def predict_proba(self, X):
        result = []
        for row in X:
            x_test = torch.from_numpy(row).float()
            softmax_score = self.model(x_test)
            softmax = torch.nn.Softmax(dim=0)
            result.append(softmax(softmax_score).data.numpy())

        return np.array(result)


if __name__ == '__main__':
    # get bag of words
    bow = loadPickle('./bag-of-words/bow_500').cluster_centers_

    # define #-dimensions
    N, D_in, H, D_out = 50, 500, 1000, 10

    # create neural network
    ann = NeuralNet(N, D_in, H, D_out)

    # get train data
    trn_encoded, labels = getTrainDataAndLabel(bow)

    # train
    ann.fit(trn_encoded, labels)
    writePickle(ann, './ann_model')
    # bow = loadPickle('./bag-of-words/bow_500').cluster_centers_
    # model = loadPickle('./ann_model')
    # img = cv2.imread('./index.jpeg')

    # encodedImg = encodeImage(img, bow)
    # print model.predict([encodedImg])
    # print model.predict_proba([encodedImg])
