from sklearn.metrics import confusion_matrix

class Evaluator(object):
    def __init__(self, classes)        
        self.classes = classes

    def get_confusion_matrix(self, model, test):
        y_pred = model.predict(test)
        self.conf_matrix = confusion_matrix(test['label'], y_pred, labels=list(set(test))

    def calculate_precision_each_class(self, model, test):        
        self.get_confusion_matrix(model, test)
        result = dict()
        for i in range(classes):
            tp = self.conf_matrix[i][i]
            total_origin = np.sum(self.conf_matrix[:, i])
            total_predict = np.sum(self.conf_matrix[i,:])
            recall = tp / total_origin
            precision = tp / total_predict
            result[i] = (recall, precision)
        return result
