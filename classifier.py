from utility import loadPickle
from sklearn.naive_bayes import MultinomialNB
from bow import encodeBatch, encodeImage
from utility import loadPickle, writePickle
from sklearn.metrics import accuracy_score
from ann import *
from extract_deepfeat import encodeImage_deepfeat, encodeBatch_deepfeat
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
        self.models.append(("softmax_deepfeat", loadPickle(
            './models/softmax/softmax_deepfeat')))
        self.models.append(("naive_bayes_deepfeat", loadPickle(
            './models/naive_bayes/multinomial_bayes_deepfeat')))
        self.models.append(("linearsvm_deepfeat",
                            loadPickle('./models/svm/linearsvm_deepfeat')))
        self.models.append(("ann_deepfeat", loadPickle(
            './models/ann/ann_3layer_deepfeat')))
        self.selected_model = self.models[0]
        self.selected_index = 0
        self.bow = bow

    def set_selected_classifier(self, index):
        self.selected_model = self.models[index]
        self.selected_index = index

    def test_accuracy(self, tst_path='./cifar-10-batches-py/test_batch'):
        print "Testing : ", self.selected_model[0]
        tst = loadPickle(tst_path)
        if self.selected_index > 4:
            tst_encoded = encodeBatch_deepfeat(tst)
        else: 
            tst_encoded = encodeBatch(tst, self.bow)
        #tst_encoded = MinMaxScaler(feature_range=(0,1)).fit(tst_encoded)
        y_pred = self.selected_model[1].predict(tst_encoded)
        score = accuracy_score(tst['labels'], y_pred) * 100
        print "Test accuracy : ", score
        return y_pred, tst, score

    def classify(self, img):
        if self.selected_index > 4:
            img_encode = encodeImage_deepfeat(img)
        else:
            img_encode = encodeImage(img, self.bow)
        img_encode.reshape(1, -1)
        y_pred = self.selected_model[1].predict_proba([img_encode])
        predict_class = self.selected_model[1].predict([img_encode])[0]
        return y_pred, predict_class


if __name__ == '__main__':
    bow = loadPickle('./bag-of-words/bow_500').cluster_centers_
    classifier = Classifier(bow)
    classifier.set_selected_classifier(0)
    classifier.test_accuracy()
