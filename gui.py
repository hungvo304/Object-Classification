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
from ann import ANN

class GUI(object):

    def __init__(self, class_labels, bow):
        self.root = Tk()
        self.setup_gui() 
        self.classifier_model = Classifier(bow)       
        self.classifier_labels = class_labels
        self.bow = bow

    def setup_gui(self):

        #self.train_label = Label(self.root, text="Train data")
        #self.train_label.grid(row=0, padx=2, pady=4)

        self.classifier_label = Label(self.root, text="Classifier")
        self.classifier_label.grid(row=1, padx=2, pady=4)
        
        #self.train_file = Text(self.root, width=30, height=1)
        #self.train_file.grid(row=0, column=1, padx=2, pady=4)
        #self.train_file.bind("<Button-1>", self.choose_directory)        

        self.classifier = ttk.Combobox(self.root)
        self.classifier['value'] = ("Naives Bayes", "SVM", "ANN", "Softmax Regression")
        self.classifier.bind("<<ComboboxSelected>>", self.set_selected_model)
        self.classifier.grid(row=1, column=1, padx=2, pady=4)
        self.classifier.current(0)

        self.button = Button(self.root, text="Image input",
                             command=self.choose_image)
        self.button.grid(row=2, padx=2, pady=4)

        self.img = ImageTk.PhotoImage("RGB", (300, 300))
        self.panel = Label(self.root, image=self.img)
        self.panel.grid(row=2, column=1)

        self.predict_class = Label(self.root, text = " ")
        self.predict_class.grid(row=3, columnspan=1)

        self.class_result = Label(self.root, text=" ")
        self.class_result.grid(row=4, columnspan=1)        

    def set_selected_model(self, event):
        self.classifier_model.set_selected_classifier(self.classifier.current())

    def choose_directory(self, event):
        folder = filedialog.askdirectory()
        self.train_file.delete("1.0", END)
        self.train_file.insert("1.0", folder.split("/")[-1])

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
        result = sorted(result, key=lambda x : x[1], reverse=True)                
        for tup in result:
            text = text + tup[0] + " : " + str(tup[1] * 100) + "\n"
        self.class_result.config(text="Classes probability : " + text, font=("Courier", 20))        
        self.predict_class.config(text="Predict class : " + self.classifier_labels[pred_class], font=("Courier", 20))
        img = ImageTk.PhotoImage(Image.open(file).resize((col/2, row/2), Image.ANTIALIAS))
        self.panel.configure(image=img)
        self.panel.image = img
       

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
