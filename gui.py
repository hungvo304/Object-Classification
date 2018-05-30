from Tkinter import *
from PIL import ImageTk, Image
import tkFileDialog as filedialog
import ttk
from utility import loadPickle
import cv2


class GUI(object):

    def __init__(self, models, class_labels):
        self.root = Tk()
        self.setup_gui()
        self.models = models
        self.selected_model = models[0]
        self.classifier_labels = class_labels

    def setup_gui(self):

        self.train_label = Label(self.root, text="Train data")
        self.train_label.grid(row=0, padx=2, pady=4)

        self.classifier_label = Label(self.root, text="Classifier")
        self.classifier_label.grid(row=1, padx=2, pady=4)

        self.train_file = Text(self.root, width=30, height=1)
        self.train_file.grid(row=0, column=1, padx=2, pady=4)
        self.train_file.bind("<Button-1>", self.choose_directory)

        self.classifier = ttk.Combobox(self.root)                       
        self.classifier['value'] = ("Naives Bayes","SVM")
        self.classifier.bind("<<ComboboxSelected>>", self.selected_model())
        self.classifier.grid(row=1, column=1, padx=2, pady=4)
        self.classifier.current(0)

        self.button = Button(self.root, text="Image input", command=self.choose_image)
        self.button.grid(row=2, padx=2, pady=4)

        self.img = ImageTk.PhotoImage("RGB", (300,300))
        self.panel = Label(self.root, image=self.img)
        self.panel.grid(row=2, column=1)    
        
        self.class_result = Label(self.root, text="Class : ")
        self.class_result.grid(row=3)

    def set_select_model(self, event):
        self.selected_model = self.models[self.classifier.get()]

    def choose_directory(self, event):
        folder = filedialog.askdirectory()
        self.train_file.delete("1.0", END)
        self.train_file.insert("1.0", folder.split("/")[-1])

    def choose_image(self):
        file = filedialog.askopenfilename()
        img = ImageTk.PhotoImage(Image.open(file))
        self.panel.configure(image=img)
        panel.image = img
        image = cv2.imread(file)
        result = model.classify(image)
        class_label = self.classifier_labels[result]
        self.class_result.configure(text="Class : " + class_label)



    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    bow = loadPickle('./bag-of-words/bow_500').cluster_centers_
    class_labels = loadPickle('./cifar-10-batches-py/batches.meta')
    print class_labels
    nb = loadPickle('./models/naive_bayes/multinomial_bayes_bow_500')
    models = [nb]
    gui = GUI(models, class_labels)
    gui.run()
