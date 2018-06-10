from utility import loadPickle
from sklearn.naive_bayes import MultinomialNB
from bow import encodeBatch, encodeImage
from utility import loadPickle, writePickle
from sklearn.metrics import accuracy_score
from ann import ANN
import torch


class Classifier(object):
    def __init__(self, bow):
        self.models = []
        self.models.append(("naive_bayes", loadPickle(
            './models/naive_bayes/multinomial_bayes_bow_500')))
        self.models.append(("svm_3rdpolynomial", loadPickle(
            './models/svm/svm_3rdpolynomial_bow_500_proba')))
        self.models.append(("svm_rbf", loadPickle(
            './models/svm/svm_rbf_bow_500_proba')))
        self.models.append(
            ("ann", loadPickle("./models/ann/ann_3layer_bow_500")))
        self.models.append(("softmax", loadPickle(
            './models/softmax/softmax_bow_500')))
        self.selected_model = self.models[0]
        self.bow = bow

    def set_selected_classifier(self, index):
        print index
        self.selected_model = self.models[index]

    def test_accuracy(self, tst_path='./cifar-10-batches-py/test_batch'):
        tst = loadPickle(tst_path)
        tst_encoded = encodeBatch(tst, self.bow)
        #tst_encoded = MinMaxScaler(feature_range=(0,1)).fit(tst_encoded)
        y_pred = self.selected_model[1].predict(tst_encoded)
        print "Test accuracy : ", accuracy_score(tst['labels'], y_pred)
        return y_pred, tst

    def classify(self, img):
        img_encode = encodeImage(img, self.bow)
        img_encode.reshape(1, -1)
        y_pred = self.selected_model[1].predict_proba([img_encode])
        predict_class = self.selected_model[1].predict([img_encode])[0]
        return y_pred, predict_class
