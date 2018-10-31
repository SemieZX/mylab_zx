# checkbutton 勾选框
import tkinter as tk

window = tk.Tk()
window.title('高光谱图像分类实验')
window.geometry('300x300')



l = tk.Label(window, bg='yellow', width=20,
             text='empty')
l.pack()


def print_selection():
    if (var1.get() == 1) & (var2.get() == 0):
        l.config(text='I love only Python ')
    elif (var1.get() == 0) & (var2.get() == 1):
        l.config(text='I love only C++')
    elif (var1.get() == 0) & (var2.get() == 0):
        l.config(text='I do not love either')
    else:
        l.config(text='I love both')

var1=tk.IntVar()
var2=tk.IntVar()

c1 = tk.Checkbutton(window, text='Python', variable=var1, onvalue=1, offvalue=0,
                    command=print_selection)
c2 = tk.Checkbutton(window, text='C++', variable=var2, onvalue=1, offvalue=0,
                    command=print_selection)

c1.pack()
c2.pack()

window.mainloop()