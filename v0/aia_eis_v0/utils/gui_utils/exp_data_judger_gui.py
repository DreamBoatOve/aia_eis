from tkinter import *
from tkinter import ttk

class exp_data_judger:
    def __init__(self):
        # self.create_gui()
        pass
    def create_gui(self):
        root = Tk()
        root.title("Data type selection")
        mainframe = ttk.Frame(root, padding="3 3 24 24")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        mainframe.columnconfigure(0, weight=1)
        mainframe.rowconfigure(0, weight=1)

        # BooleanVar的值只有int 0 和 1，0 == EIS， 1 == OCP+EIS
        data_type_var = IntVar()

        ttk.Label(mainframe, text = 'Please choose the content contained in your experiment files:').grid(column=1, row=0, columnspan=3,sticky=(W,E))
        ttk.Radiobutton(mainframe, text = 'EIS', variable= data_type_var, value= 0).grid(column=1, row=2, sticky=(W,E))
        ttk.Radiobutton(mainframe, text = 'OCP + EIS', variable= data_type_var, value= 1).grid(column=2, row=2, sticky=(W,E))
        ttk.Button(mainframe, text = 'EXIT', command=root.quit).grid(column=3, row=2, sticky=(W,E))

        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)
        root.mainloop()
        return data_type_var.get()

if __name__ == '__main__':
    # 0 == EIS， 1 == OCP + EIS
    data_type_int = exp_data_judger().create_gui()
    print(data_type_int, 'data type {}'.format(type(data_type_int)))