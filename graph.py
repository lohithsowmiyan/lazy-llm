from src.utils.ezr import o
import csv
import os
import sys
from imgcat import imgcat
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visualize(path = './output/', folder = 'single_run/' , dataset = 'auto93', show = 'All' , save_fig = False, display = False):
    data_list = []

    # Read each .txt file in the folder
    for filename in os.listdir(path + folder + dataset):
        if filename.endswith('.txt'): 
            file_path = os.path.join(path + folder + dataset, filename)
            with open(file_path, 'r') as file:
                reader = csv.reader(file, delimiter=',')  # Assuming ',' delimited files
                headers = next(reader)  
                data = {header.strip(): [] for header in headers if header.strip() != 'Var'}
                for row in reader:
                    for header, value in zip(headers, row):
                        if(header.strip() == 'Var'): continue
                        data[header.strip()].append(float(value))
                data_list.append((filename, data))

    os.makedirs(path + 'img/', exist_ok=True)

    if(show == 'All'):
        fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

        # Plot each attribute in a separate subplot
        attributes = headers[1:]  
        attributes.remove('  Var')

        for i, attribute in enumerate(attributes):
            for filename, data in data_list:
                axs[i].plot(data[headers[0].strip()], data[attribute.strip()], label=filename) 
                axs[i].set_title(attribute.strip())
                axs[i].legend()

 
        for ax in axs:
            ax.set_xlabel(headers[0].strip()) 
            ax.set_ylabel('Value')

        plt.tight_layout()

        if(save_fig):
            plt.savefig(path + 'img/' + dataset)
        if(display):
            plt.savefig(path + 'img/' + dataset +'_temp')
            im = np.asarray(Image.open(f"{path}img/{dataset}_temp.png"))
            imgcat(im, height = 70)
            os.remove(f"{path}img/{dataset}_temp.png")
            



    else:
        fig, axs = plt.subplots(1, figsize=(10, 6), sharex=True)
        for filename, data in data_list:
                axs.plot(data[headers[0].strip()], data[show], label=filename)
                axs.set_title(show)
                axs.legend()

        axs.set_xlabel('N')
        axs.set_ylabel('Value')

        plt.tight_layout()
        if(save_fig):
            plt.savefig(path + 'img/' + dataset)
        if(display):
            plt.savefig(path + 'img/' + dataset +'_temp')
            im = np.asarray(Image.open(f"{path}img/{dataset}_temp.png"))
            imgcat(im,height= 5, width = 70)
            os.remove(f"{path}img/{dataset}_temp.png")




    

if __name__ == '__main__':
  # Command Line Visualizations for a quick sneek peak at the outputs
    if(len(sys.argv) ==3):
        visualize(dataset = sys.argv[1], show = sys.argv[2], save_fig=False, display = True)
    elif(len(sys.argv) == 2):
        visualize(dataset = sys.argv[1], save_fig=False, display = True)
    else:
        visualize(save_fig=False, display = True)
