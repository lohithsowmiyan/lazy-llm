from src.utils.ezr import o
import csv
import os

def save_results(model : str = None, dataset: str = None, records : list[0] = None) -> bool:
    
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

        
    