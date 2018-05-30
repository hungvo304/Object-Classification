from Tkinter import *
import tkFileDialog as filedialog
import ttk
from utility import loadPickle


class GUI(object):

    def __init__(self):
        self.root = Tk()
        self.setup_gui()
        self.model = None

    def setup_gui(self):

        self.train_label = Label(self.root, text="Train data")
        self.train_label.grid(row=0, padx=2, pady=4)

        self.classifier_label = Label(self.root, text="Classifier")
        self.classifier_label.grid(row=1, padx=2, pady=4)

        self.train_file = Text(self.root, width=30, height=1)
        self.train_file.grid(row=0, column=1, padx=2, pady=4)
        self.train_file.bind("<Button-1>", self.choose_directory)

        self.classifier = ttk.Combobox(self.root)
        #nb = loadPickle('./model/naive_bayes/multinomial_bayes_bow_500')                
        self.classifier['value'] = ("Naives Bayes","SVM")
        self.classifier.grid(row=1, column=1, padx=2, pady=4)

        self.button = Button(self.root, text="Image input", command=self.choose_image)
        self.button.grid(row=2, padx=2, pady=4)    


    def choose_directory(self, event):
        folder = filedialog.askdirectory()
        self.train_file.delete("1.0", END)
        self.train_file.insert("1.0", folder.split("/")[-1])

    def choose_image(self):
        folder = filedialog.askopenfilename()

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    gui = GUI()
    gui.run()
