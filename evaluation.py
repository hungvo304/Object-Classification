from sklearn.metrics import confusion_matrix
from bow import encodeBatch, encodeImage
from utility import loadPickle
from navie_bayes import NaiveBayes
import numpy as np
import matplotlib.pyplot as plt

class Evaluator(object):
    def __init__(self, classes):
        self.classes = classes
        

    def get_confusion_matrix(self, model, test, classes):        
        print "Calculate confusion matrix"
        y_pred, y_test = model.test_accuracy(test)     
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

    def calculate_precision_each_class(self, conf_matrix, classes):        
        result=dict()
        for i in range(len(classes)):
            tp = conf_matrix[i][i]
            total_origin = np.sum(conf_matrix[:, i])
            total_predict = np.sum(conf_matrix[i, :])
            recall = tp / total_origin
            precision = tp / total_predict
            result[i] = (recall, precision)
        return result
    
    def draw_confusion_matrix(self, conf_matrix, classes, title="Confusion matrix"):
        conf_matrix = np.round_(conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis], 2)
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Greens)
        plt.title(title)        
        tick_index = np.arange(len(classes))
        plt.xticks(tick_index, classes, rotation=45)
        plt.yticks(tick_index, classes)
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



if __name__ == '__main__':
    print "Load model"
    nb = loadPickle('./models/naive_bayes/multinomial_bayes_bow_500')
    print "Load class labels"
    classes = loadPickle('./cifar-10-batches-py/batches.meta')['label_names']    
    print "Load bag of word"
    bow = loadPickle('./bag-of-words/bow_500').cluster_centers_

    evaluator = Evaluator(classes)
    print "Load test file"
    tests = './cifar-10-batches-py/test_batch'    
    conf_matrix = evaluator.get_confusion_matrix(nb, tests, classes)
    #result = evaluator.calculate_precision_each_class(conf_matrix, classes)    
    evaluator.draw_confusion_matrix(conf_matrix, classes)