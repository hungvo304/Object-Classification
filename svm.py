from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from bow import encodeBatch
from utility import loadPickle, writePickle
import numpy as np


def getTrainDataAndLabel(bow):

     # Train
    trn1 = loadPickle('./cifar-10-batches-py/data_batch_1')
    trn2 = loadPickle('./cifar-10-batches-py/data_batch_2')
    trn3 = loadPickle('./cifar-10-batches-py/data_batch_3')
    trn4 = loadPickle('./cifar-10-batches-py/data_batch_4')
    trn5 = loadPickle('./cifar-10-batches-py/data_batch_5')
    trn_encoded = np.concatenate((encodeBatch(trn1, bow),
                                  encodeBatch(trn2, bow),
                                  encodeBatch(trn3, bow),
                                  encodeBatch(trn4, bow),
                                  encodeBatch(trn5, bow)))
    # print trn_encoded.shape

    labels = \
        trn1['labels'] + \
        trn2['labels'] + \
        trn3['labels'] + \
        trn4['labels'] + \
        trn5['labels']

    return np.array(trn_encoded), np.array(labels)


def printOutAccuracy(model, bow, trn_path='./cifar-10-batches-py/data_batch_1',
                     tst_path='./cifar-10-batches-py/test_batch'):
    print 'Accuracy on training and testing data......'

    trn = loadPickle(trn_path)
    trn_encoded = encodeBatch(trn, bow)
    y_pred = model.predict(trn_encoded)
    print 'Train Accuracy:', 100 * accuracy_score(y_pred, trn['labels'])

    tst = loadPickle(tst_path)
    tst_encoded = encodeBatch(tst, bow)
    y_pred = model.predict(tst_encoded)
    print 'Test Accuracy: ', 100 * accuracy_score(y_pred, tst['labels'])


def train(model, bow, trn_encoded, labels):
    # Train
    print 'Training......'
    model.fit(trn_encoded, labels)

    # print accuracy
    printOutAccuracy(model, bow)

    return model


def fineTuneHyperParam(model, bow, param_grid, trn_encoded, labels):
    grid_search = GridSearchCV(
        model, param_grid, cv=5,
        verbose=2, n_jobs=-1)
    # Train
    grid_search.fit(trn_encoded, labels)

    # print accuracy
    printOutAccuracy(grid_search.best_estimator_, bow)

    return grid_search


if __name__ == '__main__':
    # # load bag of visual words
    # bow = loadPickle('./bag-of-words/bow_500').cluster_centers_

    # # load train data and labels
    # trn_encoded, labels = getTrainDataAndLabel(bow)

    # model = Pipeline((
    #     ('scaler', StandardScaler()),
    #     ('svm_clf', SVC(kernel='poly', coef0=1, C=5))
    # ))

    # param_grid = [
    #     {'svm_clf__degree': [2, 3, 4, 5]}
    # ]

    # print 'Fine-tune Model using cross-validation'
    # grid_search = fineTuneHyperParam(
    #     model, bow, param_grid, trn_encoded, labels)
    # writePickle(grid_search.best_estimator_, './models/svm/svm_poly_bow_500')
    # print grid_search

    trn_encoded = loadPickle('./deepfeat/trn_encoded')
    trn_labels = loadPickle('./deepfeat/trn_labels')
    tst_encoded = loadPickle('./deepfeat/tst_encoded')
    tst_labels = loadPickle('./deepfeat/tst_labels')

    svm = Pipeline((
        ('scaler', StandardScaler()),
        ('linear_svc', LinearSVC(C=1, loss='hinge'))
    ))
    svm.fit(trn_encoded, trn_labels)

    trn_pred = svm.predict(trn_encoded)
    print 'Train accuracy:', 100 * accuracy_score(trn_pred, trn_labels)
    tst_pred = svm.predict(tst_encoded)
    print 'Test accuracy:', 100 * accuracy_score(tst_pred, tst_labels)

    writePickle(svm, './models/svm/svm_deepfeat')
