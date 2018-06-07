from sklearn.naive_bayes import MultinomialNB
from bow import encodeBatch, encodeImage
from utility import loadPickle, writePickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import numpy as np


class NaiveBayes(object):

    def __init__(self, bow):
        self.model = Pipeline((
                            ('minmax', MinMaxScaler(feature_range= (0,1))),
                            ('nb', MultinomialNB())
                            ))
        self.model = MultinomialNB()
        self.bow = bow

    def get_train_data_and_label(self):
        print "[+] Read data from folder cifar-10-batches-py"
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
    	print trn_encoded.shape

    	labels = \
        	trn1['labels'] + \
        	trn2['labels'] + \
        	trn3['labels'] + \
        	trn4['labels'] + \
        	trn5['labels']

	return np.array(trn_encoded), np.array(labels)

    def train(self):
        trn_encoded, labels = self.get_train_data_and_label()
        print "[+] Training model"
        self.model.fit(trn_encoded, labels)

    def test_accuracy(self, tst_path='./cifar-10-batches-py/test_batch'):        
        tst = loadPickle(tst_path)
        tst_encoded = encodeBatch(tst, self.bow)
        #tst_encoded = MinMaxScaler(feature_range=(0,1)).fit(tst_encoded)
        y_pred = self.model.predict(tst_encoded)        
        print "Test accuracy : ", accuracy_score(tst['labels'], y_pred)
        return y_pred, tst

    def classify(self, img):
        img_encode = encodeImage(img, self.bow)
        img_encode.reshape(1, -1)
        y_pred = self.model.predict_proba([img_encode])        
        predict_class = self.model.predict([img_encode])[0]
        return y_pred, predict_class


if __name__ == '__main__':
    bow = loadPickle('./bag-of-words/bow_500').cluster_centers_
    bayes = NaiveBayes(bow)
    bayes.train()
    bayes.test_accuracy()
    print "Model : ", bayes.model
    writePickle(bayes.model, './models/naive_bayes/multinomial_bayes_bow_500')
