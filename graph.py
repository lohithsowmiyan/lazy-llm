from src.utils.ezr import o
import csv
import os
import sys
from imgcat import imgcat
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visualize(path = './output/', folder = 'single_run/' , dataset = 'auto93', show = 'Grid' , save_fig = False, display = False):
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

        if(save_fig):
            plt.savefig(path + 'img/' + dataset)
        if(display):
            im = np.asarray(Image.open(f"{path}img/{dataset}.png"))
            imgcat(im, height = 70)
            



    else:
        fig, axs = plt.subplots(1, figsize=(10, 15), sharex=True)
        for filename, data in data_list:
                axs.plot(data[headers[0].strip()], data[show], label=filename)  # First column as x-axis
                axs.set_title(show)
                axs.legend()

        axs.set_xlabel('N')
        axs.set_ylabel('Value')

        plt.tight_layout()
        if(save_fig):
            plt.savefig(path + 'img/' + dataset)
        if(display):
            im = np.asarray(Image.open(f"{path}img/{dataset}.png"))
            imgcat(im,height= 5, width = 70)




    

if __name__ == '__main__':
    #visualize()
    visualize(dataset = sys.argv[1], show = sys.argv[2], save_fig=True, display = True)