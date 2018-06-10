from sklearn.metrics import confusion_matrix
from bow import encodeBatch, encodeImage
from utility import loadPickle
from navie_bayes import NaiveBayes
import numpy as np
import matplotlib.pyplot as plt
from classifier import Classifier
from ann import *
import torch
    
class Evaluator(object):
    def __init__(self, classes ):
        self.classes = classes        

    def get_confusion_matrix(self, classifier, test):        
        print "Calculate confusion matrix"
        y_pred, y_test, _ = classifier.test_accuracy(test)     
        print "Test len : ", len(y_pred)
        conf_matrix = confusion_matrix(y_test['labels'], y_pred, labels=list(set(y_test['labels'])))      
        count = 0
        for x,y in zip(y_pred, y_test):
            if x ==y and x == 0:
                count +=1
        print "Count : ", count
        print "Confusion matrix : "
        print conf_matrix
        return conf_matrix

    def calculate_precision_each_class(self, conf_matrix):        
        result=dict()
        for i in range(len(self.classes)):
            tp = conf_matrix[i][i]
            total_origin = np.sum(conf_matrix[:, i])
            total_predict = np.sum(conf_matrix[i, :])
            recall = tp / total_origin
            precision = tp / total_predict
            result[i] = (recall, precision)
        return result
    
    def draw_confusion_matrix(self, conf_matrix, title="Confusion matrix"):
        conf_matrix = np.round_(conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis], 2)
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Greens)
        plt.title(title)        
        tick_index = np.arange(len(self.classes))
        plt.xticks(tick_index, self.classes, rotation=45)
        plt.yticks(tick_index, self.classes)
        fmt = 'd'       
        row = conf_matrix.shape[0]
        col = conf_matrix.shape[1]
        for i in range(row):
            for j in range(col):
                value = str(conf_matrix[i, j])
                plt.annotate(value, xy=(j, i), 
                    horizontalalignment='center',
                    verticalalignment='center')

        plt.tight_layout()        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        
    def compare_classifiers(self, classifier, test):
        model_names = [a[0] for a in classifier.models]
        accuracy = []
        for i in range(len(classifier.models)):
            classifier.set_selected_classifier(i)
            _, _, score = classifier.test_accuracy(test)
            accuracy.append(score)
        return model_names, accuracy

    def draw_accuracy_bars(self, model_names, accuracy):        
        y_pos = np.arange(len(model_names))
        plt.bar(y_pos, accuracy, align='center', alpha=0.5)
        plt.xticks(y_pos, model_names)
        plt.ylabel('Accuracy (%)')
        plt.title('Classifiers accuracy comparision')
        plt.show()

if __name__ == '__main__':
    print "Load model"
    nb = loadPickle('./models/naive_bayes/multinomial_bayes_bow_500')
    print "Load class labels"
    classes = loadPickle('./cifar-10-batches-py/batches.meta')['label_names']    
    print "Load bag of word"
    bow = loadPickle('./bag-of-words/bow_500').cluster_centers_
    classifier = Classifier(bow)
    evaluator = Evaluator(classes)
    print "Load test file"
    tests = './cifar-10-batches-py/test_batch'        
