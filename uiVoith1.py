import tkinter as tk
import random
import csv
from tkinter import *

mycolor = '#%02x%02x%02x' % (0, 58, 118)



window = tk.Tk()
window.title ( 'Failure Processing Decision' )
window.geometry( "800x800" )

img_png=tk.PhotoImage(file='img_gif.png')
label_img=tk.Label(window,image=img_png)
label_img.pack(ipadx = 65, ipady = 30)

var = tk.StringVar()
var1, var2, var3, var4, var5, var6 = tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar()

serialNr = ''
i = 0

csvFile1 = open("results_test_bench_classifier_team3.csv", 'r')
csvFile2 = open("results_failure_processing_classifier_team3.csv", 'r')
reader1 = csv.reader(csvFile1)
reader2 = csv.reader(csvFile2)

results1, results2, results3, results4 = {}, {}, {}, {}
for item in reader1:
    if reader1.line_num == 1:
        continue
    results1[item[0]] = item[1]
    results2[item[0]] = item[2]
for item in reader2:
    if reader2.line_num == 1:
        continue
    results3[item[0]] = item[1]
    results4[item[0]] = item[2]

csvFile1.close()
csvFile2.close()

serial_number1 = []
serial_number1 = list(results1.keys())
#serial_number2 = []
#serial_number2 = list(results2.keys())

test_result1 = []
test_result1 = list(results1.values())
def Hit_start():
    global serialNr
    global i
    serialNr = serial_number1[i]
    var1.set('Serial Number:  ' + serialNr)
    var2.set('Test Result:  '+ str(results1[serialNr])) 
    var5.set('Probability: ' + str(results2[serialNr])[:5])
    var3.set('Serial Number: NA')
    var4.set('Test Result: NA')
    var6.set('Probability: NA')
    
def customer_shipping():
    #read in estimated res
    #num = random.randrange(1000)
    global serialNr
    global i
    
    i +=1
    
    serialNr = serial_number1[i]
    var1.set('Serial Number:  ' + serialNr)
    var2.set('Test Result:  '+ str(results1[serialNr]))
    var5.set('Probability: ' + str(results2[serialNr])[:5])
    var3.set('Serial Number: NA')
    var4.set('Test Result: NA')
    var6.set('Probability: NA')
'''
def failure_process():
    var1.set('Serial Number:' + 'ABC')
    var2.set('0')
'''
def test_bench_decision():
    #here the second estimation is needed
    #num = random.randrange(1000)
    global i
    global serialNr
    #serialNr = serial_number2[i]
    var3.set('Serial Number:  ' + serialNr)
    var5.set('Probability: ' + str(results2[serialNr])[:5])
    var4.set('Test Result:  ' + str(results3[serialNr]))
    var6.set('Probability: ' + str(results4[serialNr])[:5])
#
def failure_process():
    global i
    global serialNr
    var3.set('Serial Number:  NA')
    var4.set('Test Result:  NA')
    var6.set('Probability: NA')
    i += 1
    serialNr = serial_number1[i]
    #Then we need to read in data of next product
    #num = random.randrange(1000)
    var1.set('Serial Number:  ' + serialNr)
    var2.set('Test Result:  '+ str(results1[serialNr]))
    var5.set('Probability:  ' + str(results2[serialNr])[:5])
    #i += 1
    #serialNr = num
    
#var1, var2, var3, var4= tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar()
var1.set('Serial Number:')
var2.set('Test Result:')
var3.set('Serial Number:')

var4.set('Test Result:')
var5.set('Probability:')
var6.set('Probability:')


l7  = tk.Label(window, text = 'Decision Model A', bg = 'white', font=('Arial', 18), width =44, height = 2)
l7.pack()
l1  = tk.Label(window, textvariable = var1, bg = 'white', font=('Arial', 18), width = 44, height = 3 )
l1.pack()
l2  = tk.Label(window, textvariable = var2, bg = 'white', font=('Arial', 18), width = 44, height = 3)
l2.pack()
l5  = tk.Label(window, textvariable = var5, bg = 'white', font=('Arial', 18), width = 44, height = 3)
l5.pack()
#l3  = tk.Label(window, textvariable = var3, bg = 'white', font=('Arial', 14), width = 35, height = 2, justify = 'left')
#l3.pack()
b0 = tk.Button(window, text='Start', fg='white',bg=mycolor,font =('Arial',14 ),width =55, height = 3,command = Hit_start)
b0.pack()
b1 = tk.Button(window, text='send the product to Customer Shipping', fg='white',bg=mycolor, font =('Arial',14 ),width =55, height = 3,command = customer_shipping)
b1.pack()
b2 = tk.Button(window, text='send to Test Bench', font =('Arial',14 ), fg='white',bg=mycolor ,width =55, height = 3,command = test_bench_decision)
b2.pack()


'''
b1 = tk.Button(window, text=var, command = customer_shipping)
b1.pack()

b2 = tk.Button(window, text='Test Bench', command = test bench)
b2.pack()
'''

top = tk.Toplevel(window)
top.geometry("800x800")
top.title( 'Failure Processing Decision' )
#frame = tk.Frame(window)

img_png2=tk.PhotoImage(file='img_gif.png')
label_img2=tk.Label(top,image=img_png2,justify=tk.RIGHT)
label_img2.pack(ipadx = 65, ipady = 30)

l8  = tk.Label(top, text = 'Decision Model B', bg = 'white', font=('Arial', 18), width =44, height = 2)
l8.pack()
l3  = tk.Label(top, textvariable = var3, bg = 'white', font=('Arial', 18), width = 44, height = 3)
l3.pack()
l4  = tk.Label(top, textvariable = var4, bg = 'white', font=('Arial', 18), width = 44, height = 3)
l4.pack()
l6  = tk.Label(top, textvariable = var6, bg = 'white', font=('Arial', 18), width = 44, height = 3)
l6.pack()

btn1 = tk.Button(top, text='send the product to Customer Shipping',font =('Arial',14 ), fg='white',bg=mycolor,width =55, height = 3, command = customer_shipping)
btn1.pack()
btn2 = tk.Button(top, text='send to Failure Processing',font =('Arial',14 ), fg='white',bg=mycolor,width =55, height = 3, command = failure_process)
btn2.pack()

#btnAdd = Button(top, text="Add User", font="Helvetica 20 bold", compound=TOP)

window.mainloop()