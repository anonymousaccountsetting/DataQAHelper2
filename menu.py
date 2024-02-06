import tkinter as tk
from exploredataregression import LinearRegressionApp
from trainmodelregression import LinearRegressionModelApp
from exploredataclassifier import ClassifierApp

class MenuApp:
    def __init__(self, root):
        self.root = root
        self.root.title("What model are you looking for?")
        self.create_widgets()

    def create_widgets(self):
        self.button1 = tk.Button(self.root, text="1. A regression model for exploring data",
                                 command=self.launch_explore_data_regression)
        self.button1.pack(padx=20, pady=10, fill=tk.X)

        self.button2 = tk.Button(self.root, text="2. A regression model for training a good model",
                                 command=self.launch_train_model_regression)
        self.button2.pack(padx=20, pady=10, fill=tk.X)

        self.button3 = tk.Button(self.root, text="3. A classifier model for exploring data",
                                 command=self.launch_explore_data_classifier)
        self.button3.pack(padx=20, pady=10, fill=tk.X)

    def launch_explore_data_regression(self):
        app_root = tk.Toplevel(self.root)
        app = LinearRegressionApp(app_root)

    def launch_train_model_regression(self):
        app_root = tk.Toplevel(self.root)
        app = LinearRegressionModelApp(app_root)

    def launch_explore_data_classifier(self):
        app_root = tk.Toplevel(self.root)
        app = ClassifierApp(app_root)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("400x200")
    app = MenuApp(root)
    root.mainloop()
