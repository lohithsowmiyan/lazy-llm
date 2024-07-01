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

def visualize(path = './output/', folder = 'single_run/' , dataset = 'auto93', show = 'Grid' ,save_fig = False):
    data_list = []

    # Read each .txt file in the folder
    for filename in os.listdir(path + folder + dataset):
        if filename.endswith('.txt'): 
            file_path = os.path.join(path + folder + dataset, filename)
            with open(file_path, 'r') as file:
                reader = csv.reader(file, delimiter=',')  # Assuming tab-delimited files
                headers = next(reader)  # Read the headers
                data = {header.strip(): [] for header in headers if header.strip() != 'Var'}
                for row in reader:
                    for header, value in zip(headers, row):
                        if(header.strip() == 'Var'): continue
                        data[header.strip()].append(float(value))
                data_list.append((filename, data))

    os.makedirs(path + 'img/', exist_ok=True)

    if(show == 'Grid'):
        fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

        # Plot each attribute in a separate subplot
        attributes = headers[1:]  # Assuming the first column is used as x-axis
        attributes.remove('  Var')

        for i, attribute in enumerate(attributes):
            for filename, data in data_list:
                axs[i].plot(data[headers[0].strip()], data[attribute.strip()], label=filename)  # First column as x-axis
                axs[i].set_title(attribute.strip())
                axs[i].legend()

 
        for ax in axs:
            ax.set_xlabel(headers[0].strip())  # Set the x-axis label from the header
            ax.set_ylabel('Value')

        plt.tight_layout()
        plt.savefig(path + 'img/' + dataset) if save_fig == True else None
        plt.show()



    else:
        fig, axs = plt.subplots(1, figsize=(10, 15), sharex=True)
        for filename, data in data_list:
                axs.plot(data[headers[0].strip()], data[show], label=filename)  # First column as x-axis
                axs.set_title(show)
                axs.legend()

        axs.set_xlabel('N')
        axs.set_ylabel('Value')

        plt.tight_layout()
        plt.savefig(path + 'img/' + dataset) if save_fig == True else None
        plt.show()




    

if __name__ == '__main__':
    #visualize()
    visualize(dataset = sys.argv[1], show = sys.argv[2], save_fig=True)





        
        
    