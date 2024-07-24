import csv

with open('file.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)