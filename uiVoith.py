import tkinter as tk
import random
import csv

window = tk.Tk()
window.title ( 'Predictive Analytics' )
window.geometry( "800x800" )

filename1 = './fakeData/results_test_bench_classifier_team3.csv'
filename2 = './fakeData/results_failure_processing_classifier_team3.csv'

var = tk.StringVar()
var1, var2, var3, var4, var5, var6 = tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar()

serialNr = ''
i = 0

csvFile1 = open(filename1, 'r')
csvFile2 = open(filename2, 'r')
reader1 = csv.reader(csvFile1)
reader2 = csv.reader(csvFile2)

results1, results2 = {}, {}
for item in reader1:
    if reader1.line_num == 1:
        continue
    results1[item[0]] = item[1]
for item in reader2:
    if reader2.line_num == 1:
        continue
    results2[item[0]] = item[1]

csvFile1.close()
csvFile2.close()

serial_number1 = []
serial_number1 = list(results1.keys())
#serial_number2 = []
#serial_number2 = list(results2.keys())

test_result1 = []
test_result1 = list(results1.values())

def customer_shipping():
    #read in estimated res
    #num = random.randrange(1000)
    global serialNr
    global i
    
    serialNr = serial_number1[i]
    var1.set('Serial Number:' + serialNr)
    var2.set(str(test_result1[i]))
    i += 1
    var3.set('Serial Number: NA')
    var4.set('Test Result: NA')
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
    var3.set('Serial Number:' + serialNr)
    var4.set('Test Result:' + str(results2[serialNr]))

def failure_process():
    global serialNr
    var3.set('Serial Number: NA')
    var4.set('Test Result: NA')
    #Then we need to read in data of next product
    #num = random.randrange(1000)
    #var1.set('Serial Number:' + str(num))
    #var2.set(str(num))
    #serialNr = num
    
var1, var2, var3, var4, var5, var6 = tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar()
var1.set('Start')
var2.set('serial_number')
var3.set('serial_number')

var4.set('test_result')
var5.set('Failure')
var6.set('Failure')

l1  = tk.Label(window, textvariable = var1, bg = 'white', font=('Arial', 14), width = 35, height = 2)
l1.pack()
l2  = tk.Label(window, textvariable = var2, bg = 'white', font=('Arial', 14), width = 35, height = 2, justify = 'left')
l2.pack()
#l3  = tk.Label(window, textvariable = var3, bg = 'white', font=('Arial', 14), width = 35, height = 2, justify = 'left')
#l3.pack()
 
b1 = tk.Button(window, text='S', command = customer_shipping)
b1.pack()
b2 = tk.Button(window, text='T', command = test_bench_decision)
b2.pack()
'''
b1 = tk.Button(window, text=var, command = customer_shipping)
b1.pack()

b2 = tk.Button(window, text='Test Bench', command = test bench)
b2.pack()
'''

top = tk.Toplevel(window)
top.geometry("800x800")
top.title( 'Top' )
#frame = tk.Frame(window)

l3  = tk.Label(top, textvariable = var3, bg = 'white', font=('Arial', 14), width = 35, height = 2)
l3.pack()
l4  = tk.Label(top, textvariable = var4, bg = 'white', font=('Arial', 14), width = 35, height = 2, justify = 'left')
l4.pack()
btn1 = tk.Button(top, text='S', command = customer_shipping)
btn1.pack()
btn2 = tk.Button(top, text='F', command = failure_process)
btn2.pack()

#btnAdd = Button(top, text="Add User", font="Helvetica 20 bold", compound=TOP)

window.mainloop()