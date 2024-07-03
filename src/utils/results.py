from src.utils.ezr import o
import csv
import os
import sys
import matplotlib.pyplot as plt

def save_results_csv(model : str = None, dataset: str = None, records : list[0] = None) -> bool:
    
    dataset = dataset[dataset.rfind('/') + 1: -4]
    directory = './output/single_run/'+dataset+'/'
    os.makedirs(directory, exist_ok=True)
    filename = directory + model+'.csv'
    with open(filename, 'w', newline = '') as csvfile:
        csvwrite = csv.writer(csvfile, delimiter = ',')

        for record in records:
            row = [record.N, record.Mu, record.Sd, record.Var]
            print(row)
            csvwrite.writerow(row)

    return True

def save_results_txt(model : str = None, dataset: str = None, records : list[0] = None) -> bool:
    dataset = dataset[dataset.rfind('/') + 1: -4]
    directory = './output/single_run/'+dataset+'/'
    os.makedirs(directory, exist_ok=True)
    filename = directory + model+'.txt'
    def format_row(row):
        return ', '.join([f' {item}' for item in row])

    with open(filename, 'w', newline = '') as txtfile:
        header = ['N', 'Mu', 'Sd', 'Var', 'Curd2h', 'Budget']
        txtfile.write(format_row(header) + '\n')

        for record in records:
            row = [record.N, record.Mu, record.Sd, record.Var, record.Curd2h, record.Budget]
            print(row)
            txtfile.write(format_row(row) + '\n')

    return True







        
        
    