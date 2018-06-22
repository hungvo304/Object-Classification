from Tkinter import *
from PIL import ImageTk, Image
import tkFileDialog as filedialog
import ttk
from utility import loadPickle
import cv2
from navie_bayes import NaiveBayes
from bow import encodeImage
import numpy as np
from classifier import Classifier
from ann import *
from evaluation import Evaluator

custom_font = ("Courier", 15)

class GUI(object):

    def __init__(self, class_labels, bow):
        self.root = Tk()
        self.setup_gui()
        self.classifier_model = Classifier(bow)
        self.classifier_labels = class_labels
        self.bow = bow
        self.evaluator = Evaluator(class_labels)

    def setup_gui(self):

        self.classifier_label = Label(self.root, text="Classifier", font=custom_font)
        self.classifier_label.grid(row=1, padx=2, pady=4)

        self.classifier = ttk.Combobox(self.root, width=40)
        self.classifier['value'] = (
            "Naives Bayes", "SVM 3rd Polynomial", "SVM RBF", "ANN", "Softmax Regression", "Softmax Regression Deep Feature", "Naives Bayes Deep Feature", "Linear SVM Deep Feature", "ANN Deep Feature")
        self.classifier.bind("<<ComboboxSelected>>", self.set_selected_model)
        self.classifier.grid(row=1, column=1, padx=2, pady=4)
        self.classifier.current(0)

        self.button = Button(self.root, text="Image input",
                             command=self.choose_image)
        self.button.grid(row=2, padx=2, pady=4)

        self.img = ImageTk.PhotoImage("RGB", (300, 300))
        self.panel = Label(self.root, image=self.img)
        self.panel.grid(row=2, column=1)

        self.predict_class = Label(self.root, text=" ", font=custom_font)
        self.predict_class.grid(row=3, columnspan=1)

        self.class_result = Label(self.root, text=" ")
        self.class_result.grid(row=4, columnspan=1)

        self.list_progress_bars = list()
        for i in range(10):
            class_name = Label(
                self.root, text="Class name #" + str(i), font=custom_font)
            class_name.grid(row=5 + i, column=0)

            class_bar = ttk.Progressbar(
                self.root, orient="horizontal", length=200, mode="determinate")
            class_bar["maximum"] = 100
            class_bar["value"] = 0
            class_bar.grid(row=5 + i, column=1)

            percentage_text = "0%"
            percentage_label = Label(self.root, text=percentage_text, font=custom_font)
            percentage_label.grid(row=5 + i, column=2)
            self.list_progress_bars.append(
                (class_name, class_bar, percentage_label))

#        self.evaluate_Label = Label(
#            self.root, text="Evaluate model", font=("Courier", 16))
#        self.evaluate_Label.grid(row=17)

#        self.test_selection = Text(self.root, width=40, height=1)
#        self.test_selection.grid(row=17, column=1)
#        self.test_selection.bind("<Button-1>", self.choose_file)
#
#        self.evalute_button = Button(
#            self.root, text="Evaluate", command=self.show_confusion_matrix)
#        self.evalute_button.grid(row=17, column=2)

#    def show_confusion_matrix(self):
#        path = self.test_selection.get("1.0", 'end-1c')
#        conf_matrix = self.evaluator.get_confusion_matrix(
#            self.classifier_model, path)
#        self.evaluator.draw_confusion_matrix(
#            conf_matrix, title=self.classifier_model.selected_model[0] + " confusion matrix")

    def set_selected_model(self, event):
        self.classifier_model.set_selected_classifier(
            self.classifier.current())

    def choose_file(self, event):
        file = filedialog.askopenfilename()
        self.test_selection.delete("1.0", END)
        self.test_selection.insert("1.0", file)

    def choose_image(self):
        file = filedialog.askopenfilename()
        print "[!] Read image", file
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        row, col = image.shape[0], image.shape[1]
        y_pred, pred_class = self.classifier_model.classify(image)
        text = "\n"
        result = []
        for i in range(len(y_pred[0])):
            result.append((self.classifier_labels[i], y_pred[0][i]))
        result = sorted(result, key=lambda x: x[1], reverse=True)

        for i in range(len(result)):
            self.change_percentage_bar(
                self.list_progress_bars[i], result[i][0], result[i][1])

        self.class_result.config(
            text="Classes probability : " + text, font=custom_font)
        self.predict_class.config(
            text="Predict class : " + self.classifier_labels[pred_class], font=custom_font)
        img = ImageTk.PhotoImage(Image.open(file))  # .resize(
        #(col / 2, row / 2), Image.ANTIALIAS))
        self.panel.configure(image=img)
        self.panel.image = img

    def change_percentage_bar(self, progress_bar, label, percentage):
        class_name = progress_bar[0]
        class_name.config(text=label, font=custom_font)

        class_bar = progress_bar[1]
        class_bar["maximum"] = 100
        class_bar["value"] = percentage * 100

        percentage_text = str(percentage * 100) + "%"
        percentage_label = progress_bar[2]
        percentage_label.config(text=percentage_text, font=custom_font)

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    print "[+] Read bag of word"
    bow = loadPickle('./bag-of-words/bow_500').cluster_centers_
    class_labels = loadPickle(
        './cifar-10-batches-py/batches.meta')['label_names']
    print "[+] Load set of labels : ", class_labels
    gui = GUI(class_labels, bow)
    gui.run()
