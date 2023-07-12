import tkinter as tk
from tkinter import ttk

import json

# open the file in read mode
with open("data.json", "r") as file:
    # read the dictionary from the file
    data = json.load(file)



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

        self.labels = [ttk.Label(self.frame, text=f"{key}:") for key in data.keys()]
        for index, label in enumerate(self.labels):
            label.grid(row=index, column=0)
            
        
        self.comboboxes = [ttk.Combobox(self.frame, values=value, width=10) for value in data.values()]
        for index, combobox in enumerate(self.labels):
            combobox.grid(row=index, column=1)

        # self.label1 = ttk.Label(self.frame, text="Enter value 1:")
        # self.label1.grid(row=0, column=0)

        # self.entry1 = ttk.Entry(self.frame, width=10)
        # self.entry1.grid(row=0, column=1)

        # self.label2 = ttk.Label(self.frame, text="Select a category:")
        # self.label2.grid(row=20, column=0)

        



        self.button = ttk.Button(self.frame, text="Calculate", command=self.calculate_sum)
        self.button.grid(row=2, column=0, columnspan=2)

        self.result = ttk.Label(self.frame, text="")
<<<<<<< HEAD
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
=======
        self.result.grid(row=3, column=0, columnspan=2)
>>>>>>> parent of d23042a (GUI working. Calulus todo)

    def calculate_sum(self):
        
        value1 = int(self.entry1.get())
        value2 = self.combobox.get()
        result = value1 + int(value2[-1])
        
        
        self.result.configure(text="Result: " + str(result))

root = tk.Tk()
# set the size of the window
# root.geometry("400x200")
my_gui = MyGUI(root)
root.mainloop()