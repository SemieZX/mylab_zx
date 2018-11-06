from tkinter import *
fwin = Tk()
fwin.title("框架窗口")
fwin.geometry("400x400")

fram = Frame(fwin)
fram.pack()

left = Frame(fram, width=200, height=200, bg='blue')
left.pack(side='left')

right = Frame(fram, width=200, height=200)
right.pack(side='right')

top = Frame(fram, width=200, height=200)
top.pack(side='top')

bottom = Frame(fram, width=200, height=200)
bottom.pack(side='bottom')


lable_left = Label(left, text='左边')
lable_left.pack()

lable_right = Label(right, text='右边')
lable_right.pack()

label_top = Label(top, text='上边')
label_top.pack()

label_bottom = Label(bottom, text='下边')
label_bottom.pack()


fwin.mainloop()