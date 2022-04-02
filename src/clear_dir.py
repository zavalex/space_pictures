import os
import sys
import csv

def main():
    args = sys.argv[1:]
    PATH = args[0]
    counter = 0
    with open(PATH, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if int(row[1]) != 0 and int(row[1]) != 2:
                os.remove('../test_files/images/10/'+row[0]) 
                counter += 1
                print(', '.join(row))
    print(str(counter)+' pics removed.')

if __name__ == "__main__":
    main()