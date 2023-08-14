import pandas as pd
import tkinter as tk
from tkinter import ttk

import json
from joblib import load

try:
    from dataManipulation import userInputFeatures, getExtraFeatures, DataManipulation
except ModuleNotFoundError:
    from app.dataManipulation import userInputFeatures, getExtraFeatures, DataManipulation

# open the file in read mode
try:
    file = open("data.json", "r")
    home = "../"
except FileNotFoundError:
    file = open("app/data.json", "r")
    home = ""

# read the dictionary from the file
data = json.load(file)
file.close()

generalDF = pd.concat([pd.read_csv(f'{home}archive/train.csv'), 
                          pd.read_csv(f'{home}archive/test.csv')])



class MyGUI:


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


        # Categorical Features Input
        self.categoricalUserInputFeatures = [key for key in data["categorical"].keys() if key in userInputFeatures]
        self.labels = [ttk.Label(self.frame, text=f"{key}:") for key in self.categoricalUserInputFeatures]
        for index, label in enumerate(self.labels):
            label.grid(row=index, column=0)
        for value in data["categorical"].values():
            value.sort()
        self.comboboxes = [ttk.Combobox(self.frame, values=value) for value in 
                           [data["categorical"][feature] for feature in self.categoricalUserInputFeatures]]
        for index, combobox in enumerate(self.comboboxes):
            combobox.grid(row=index, column=1)

        # Numerical Features Input
        self.numericalUserInputFeatures = [key for key in data["numerical"].keys() if key in userInputFeatures]
        self.labels2 = [ttk.Label(self.frame, text=f"{key}:") for key in self.numericalUserInputFeatures]
        for index, label in enumerate(self.labels2):
            label.grid(row=len(self.labels)+index, column=0)

        self.sliders = [tk.Scale(self.frame, from_=min(value), to=max(value), orient=tk.HORIZONTAL) for value in 
                        [data["numerical"][feature] for feature in self.numericalUserInputFeatures]]
        for index, slider in enumerate(self.sliders):
            slider.grid(row=len(self.labels)+index, column=1)
        



        self.button = ttk.Button(self.frame, text="Calculate", command=self.calculate_sum)
        self.button.grid(row=len(self.labels)+len(self.labels2)+1, column=0, columnspan=2)

        self.result = ttk.Label(self.frame, text="")
        self.result.grid(row=len(self.labels)+len(self.labels2)+2, column=0, columnspan=2)
        
        self.dataManipulation = DataManipulation(home)
        

    def generate_df(self, data):
        df = pd.DataFrame(columns = generalDF.columns)
        for combobox, key in zip(self.comboboxes, self.categoricalUserInputFeatures):
            df.loc[0, key] = combobox.get()
        for slider, key in zip(self.sliders, self.numericalUserInputFeatures):
            df.loc[0, key] = slider.get()
        
        df = getExtraFeatures(df, generalDF)
        return df


    def calculate_sum(self):
        generated_df = self.generate_df(data)
        predictedPrice = self.dataManipulation.predict(generated_df)
        try:
            self.label.destroy()
        except AttributeError:
            pass
        self.label = ttk.Label(self.frame, text=f'Precio: {predictedPrice:.0f}')
        self.label.grid(row=len(self.labels)+len(self.labels2)+3, column=0)
        



root = tk.Tk()
# set the size of the window
root.geometry("800x600")
my_gui = MyGUI(root)
root.mainloop()