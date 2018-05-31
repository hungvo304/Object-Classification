from sklearn.linear_model import LogisticRegression
from svm import getTrainDataAndLabel, train
from utility import loadPickle


if __name__ == '__main__':
    bow = loadPickle('./bag-of-words/bow_1000').cluster_centers_

    trn_encoded, labels = getTrainDataAndLabel(bow)

    softmax_reg = LogisticRegression(
        multi_class='multinomial', solver='lbfgs', C=10)
    train(softmax_reg, bow, trn_encoded, labels)
