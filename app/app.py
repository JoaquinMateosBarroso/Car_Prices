import tkinter as tk
from tkinter import ttk
import pandas as pd

import json
from joblib import load

# open the file in read mode
with open("data.json", "r") as file:
    # read the dictionary from the file
    data = json.load(file)





class MyGUI:
    def prepare_models(self):
        # load models from the file
        self.preparation_pipeline = load('../models/preparation_pipeline.joblib')
        self.pca = load("../models/pca.joblib")
        self.final_model = load("../models/final_model.joblib")
        
    def __init__(self, master):
        self.master = master
        master.title("Predict Car Prices")

        self.style = ttk.Style()
        self.style.configure('TLabel', font=('Arial', 14), foreground='black')
        self.style.configure('TButton', font=('Arial', 14), foreground='white', background='blue')
        self.style.configure('TCombobox', font=('Arial', 14))

        # create a frame to hold the widgets
        self.frame = ttk.Frame(master, padding=20)
        self.frame.pack()

        self.labels = [ttk.Label(self.frame, text=f"{key}:") for key in data["categorical"].keys()]
        for index, label in enumerate(self.labels):
            label.grid(row=index, column=0)
        
        self.comboboxes = [ttk.Combobox(self.frame, values=value) for value in data["categorical"].values()]
        for index, combobox in enumerate(self.comboboxes):
            combobox.grid(row=index, column=1)


        self.labels2 = [ttk.Label(self.frame, text=f"{key}:") for key in data["numerical"].keys()]
        for index, label in enumerate(self.labels2):
            label.grid(row=len(self.labels)+index, column=0)

        self.sliders = [tk.Scale(self.frame, from_=min(value), to=max(value), orient=tk.HORIZONTAL) for value in data["numerical"].values()]
        for index, slider in enumerate(self.sliders):
            slider.grid(row=len(self.labels)+index, column=1)
        



        self.button = ttk.Button(self.frame, text="Calculate", command=self.calculate_sum)
        self.button.grid(row=len(self.labels)+len(self.labels2)+1, column=0, columnspan=2)

        self.result = ttk.Label(self.frame, text="")
        self.result.grid(row=len(self.labels)+len(self.labels2)+2, column=0, columnspan=2)
        
        self.prepare_models()

    def generate_df(self, data):
        df = pd.DataFrame(columns = list(data["categorical"].keys()) + list(data["numerical"].keys()))
        for combobox, key in zip(self.comboboxes, data["categorical"].keys()):
            df.loc[0, key] = combobox.get()
        for slider, key in zip(self.sliders, data["numerical"].keys()):
            df.loc[0, key] = slider.get()
        return df
        
        
    def predict(self, df):
        X = self.preparation_pipeline.transform(df)
        X = self.pca.transform(X)
        Y = self.final_model.predict(X)
        return Y[0]

    def calculate_sum(self):
        print(self.predict(self.generate_df(data)))
        
        
        


root = tk.Tk()
# set the size of the window
root.geometry("800x600")
my_gui = MyGUI(root)
root.mainloop()