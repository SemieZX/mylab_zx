# Scale 部件
import tkinter as tk

window = tk.Tk()
window.title('高光谱图像分类实验')
window.geometry('300x300')



l = tk.Label(window, bg='yellow', width=20,
             text='empty')
l.pack()


def print_selection(v):
    l.config(text='you have selected' + v)

s = tk.Scale(window, label='try me', from_=5,to=11,
             orient=tk.HORIZONTAL,length=200,
             showvalue=0, tickinterval=3, resolution=0.01,
             command=print_selection)

s.pack()

window.mainloop()