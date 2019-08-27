from numpy import genfromtxt
import csv
import numpy as np
# 'Data/assembly_training.csv'
def getarray(path):
    return genfromtxt(path, delimiter=',')
# 'Data/field_failure_training.csv'
def csvfailure(path='Data/field_failure_training.csv'):
    listofID = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count==0:
                line_count += 1
                continue
            else:
                listofID.append(row[0])
                line_count += 1
        return listofID

def csvAssembly(path):
    listofID = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count==0:
                line_count += 1
                continue
            else:
                listofID.append(row[:])
                line_count += 1
        return listofID

def getOrderedValidID(stage):
    failureID = csvfailure()
    assembly_ID = []
    bench_ID = []
    if stage=='assembly':
        FID = []
        with open('data/assembly_validate.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count==0:
                    line_count += 1
                    continue
                else:
                    if row[1] in failureID:
                        FID.append(row[1])
                    else:
                        assembly_ID.append(row[1])
                line_count += 1
            assembly_ID.append(FID)
            return assembly_ID
    else:
        FID = []
        with open('data/test_bench_validate.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count==0:
                    line_count += 1
                    continue
                else:
                    if row[3] in failureID:
                        FID.append(row[3])
                    else:
                        bench_ID.append(row[3])
                line_count += 1
            bench_ID.append(FID)
            return bench_ID

def getTextID(path, stage):
    # failureID = csvfailure()
    assembly_ID = []
    bench_ID = []
    # if stage=='assembly':
    #     FID = []
    if stage=='assembly':
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count==0:
                    line_count += 1
                    continue
                else:
                    assembly_ID.append(row[1])
                line_count += 1
            return assembly_ID
    else:
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count==0:
                    line_count += 1
                    continue
                else:
                    bench_ID.append(row[3])
                line_count += 1
            return bench_ID

def splitLabelData():
    good = []
    bad = []
    listOfID = csvfailure('Data/field_failure_validate.csv')
    listAssembly = csvAssembly('Data/90.csv')
    for i in range(0, len(listAssembly)):
        print(i)
        if listAssembly[i][3] in listOfID:
            bad.append(listAssembly[i][4:])
        else:
            good.append(listAssembly[i][4:])
    return good, bad
    # listOfID = np.array(listOfID)[:, 0]
    # print(listAssembly[0])
    # assemblyData = getarray('Data/assembly_training.csv')
    # listOfID = list(map(float, listOfID))

