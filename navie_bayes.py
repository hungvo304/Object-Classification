from sklearn.naive_bayes import MultinomialNB
from bow import encodeBatch, encodeImage
from utility import loadPickle, writePickle
from sklearn.metrics import accuracy_score


class NaiveBayes(object):

    def __init__(self, bow):
        self.model = MultinomialNB()
        self.bow = bow

    def get_train_data_and_label(self):
        print "[+] Read data from folder cifar-10-batches-py"
        trn1 = loadPickle('./cifar-10-batches-py/data_batch_1')
        trn2 = loadPickle('./cifar-10-batches-py/data_batch_2')
        trn3 = loadPickle('./cifar-10-batches-py/data_batch_3')
        trn4 = loadPickle('./cifar-10-batches-py/data_batch_4')
        trn5 = loadPickle('./cifar-10-batches-py/data_batch_5')
        trn_encoded = \
            encodeBatch(trn1, bow) + \
            encodeBatch(trn2, bow) + \
            encodeBatch(trn3, bow) + \
            encodeBatch(trn4, bow) + \
            encodeBatch(trn5, bow)

        labels = \
            trn1['labels'] + \
            trn2['labels'] + \
            trn3['labels'] + \
            trn4['labels'] + \
            trn5['labels']
        return trn_encoded, labels

    def train(self):
        trn_encoded, labels = self.get_train_data_and_label()
        print "[+] Training model"
        self.model.fit(trn_encoded, labels)

    def test_accuracy(self, tst_path='./cifar-10-batches-py/test_batch'):
        print "[!] Calculating accuracy : "
        tst = loadPickle(tst_path)
        tst_encoded = encodeBatch(tst, self.bow)
        y_pred = self.model.predict(tst_encoded)
        print '[+] Test Accuracy', accuracy_score(y_pred, tst['labels']) * 100

    def classify(self, img):
        img_encode = encodeImage(img, self.bow)
        img_encode.reshape(1, -1)
        y_pred = self.model.predict([img_encode])[0]
        print "[+] Image class : ", y_pred
        return y_pred


if __name__ == '__main__':
    bow = loadPickle('./bag-of-words/bow_500').cluster_centers_
    bayes = NaiveBayes(bow)
    bayes.train()
    bayes.test_accuracy()
    writePickle(bayes, './models/naive_bayes/multinomial_bayes_bow_500')
