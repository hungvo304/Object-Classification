from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from svm import getTrainDataAndLabel, train
from utility import loadPickle, writePickle
from extract_deepfeat import extractDeepFeat


def train_deepfeat(model, trn_encoded, trn_labels, tst_encoded, tst_labels):
    # Train
    print 'Training......'
    model.fit(trn_encoded, labels)

    # print accuracy
    trn_pred = model.predict(trn_encoded)
    print 'Train Accuracy:', 100 * accuracy_score(trn_pred, trn_labels)

    tst_pred = model.predict(tst_encoded)
    print 'Test Accuracy:', 100 * accuracy_score(tst_pred, tst_labels)

    return model


def fineTuneHyperParam_deepfeat(model, param_grid, trn_encoded, trn_labels, tst_encoded, tst_labels):
    grid_search = GridSearchCV(
        model, param_grid, cv=10,
        verbose=2, n_jobs=-1)
    # Train
    grid_search.fit(trn_encoded, trn_labels)

    # print accuracy
    trn_pred = grid_search.best_estimator_.predict(trn_encoded)
    print 'Train Accuracy:', 100 * accuracy_score(trn_pred, trn_labels)

    tst_pred = grid_search.best_estimator_.predict(tst_encoded)
    print 'Test Accuracy:', 100 * accuracy_score(tst_pred, tst_labels)

    return grid_search


if __name__ == '__main__':
    # bow = loadPickle('./bag-of-words/bow_1000').cluster_centers_

    # trn_encoded, labels = getTrainDataAndLabel(bow)
    # trn_encoded, labels = extractDeepFeat()
    trn_encoded = loadPickle('./deepfeat/trn_encoded')
    trn_labels = loadPickle('./deepfeat/trn_labels')
    tst_encoded = loadPickle('./deepfeat/tst_encoded')
    tst_labels = loadPickle('./deepfeat/tst_labels')

    softmax_reg = LogisticRegression(
        multi_class='multinomial', solver='lbfgs')
    param_grid = [
        {'C': [10]}
    ]
    grid_search = fineTuneHyperParam_deepfeat(
        softmax_reg, param_grid, trn_encoded, trn_labels, tst_encoded, tst_labels)
    # softmax_reg.fit(trn_encoded, trn_labels)
    # softmax_reg = loadPickle('./models/softmax/softmax_deepfeat')
    # writePickle(softmax_reg, './models/softmax/softmax_deepfeat')

    writePickle(grid_search, './gridsearch/gridsearch_softmax_deepfeat')
