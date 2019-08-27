import csv
import numpy as np


def main():
    filename = 'data/test_bench_validate.csv'
    with open(filename, newline='') as csvfile:
        testbench_list = list(csv.reader(csvfile, delimiter=',', quotechar='|'))

    filename = 'data/assembly_validate.csv'
    with open(filename, newline='') as csvfile:
        assembly_list = list(csv.reader(csvfile, delimiter=',', quotechar='|'))

    filename = 'data/field_failure_validate.csv'
    with open(filename, newline='') as csvfile:
        failure_list = list(csv.reader(csvfile, delimiter=',', quotechar='|'))

    for row in testbench_list:
        for r1 in assembly_list:
            if row[3] == r1[1]:
                row.append(r1[2:])
        for r2 in failure_list:
            if r2[1] == row[3]:
                with open("data/90_validate_neg.csv", 'a', newline='') as outputcsvfile:
                    writer1 = csv.writer(outputcsvfile, delimiter=',')
                    writer1.writerow(row)

                with open("data/90_validate_pos.csv", 'a', newline='') as outputcsvfile:
                    writer1 = csv.writer(outputcsvfile, delimiter=',')
                    writer1.writerow(row)


if __name__ == '__main__':
    main()
    print("End")