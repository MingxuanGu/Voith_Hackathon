for row in allSampleList:
            if row[1] in badSerialNrList:
                allSerialNrList.append([row[1], 1])
                with open(outputFileName, 'a', newline='') as outputcsvfile:
                    writer1 = csv.writer(outputcsvfile, delimiter=',')
                    writer1.writerow(row + ['0'])
                #print("No!")
            else:
                with open(outputFileName, 'a', newline='') as outputcsvfile:
                    writer2 = csv.writer(outputcsvfile, delimiter=',')
                    writer2.writerow(row + ['1'])
                allSerialNrList.append([row[1], 0])
                #print("Yes!")