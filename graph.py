from src.utils.ezr import o,d2h
import csv
import os
import sys
from imgcat import imgcat
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.utils.pca import PCA
import random

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
            imgcat(im,height= 10, width = 90)
            os.remove(f"{path}img/{dataset}_temp.png")


def visualize2(dataset: str, graphs):

    num_policies = len(graphs)
    num_samples = 5

    # Create a figure with subplots arranged in a grid
    fig, axs = plt.subplots(num_samples, num_policies, figsize=(4 * num_policies, 4 * num_samples))
    for col_idx, (policy, data) in enumerate(graphs.items()):
        sampled_data = random.sample(data, k=num_samples)
        for row_idx, d in enumerate(sampled_data):
            [i, most, best, rest, done] = d
            s = sorted(i.rows, key = lambda r : d2h(i,r))
            n = len(i.rows)
            bottom_5th_index = int(0.05 * n)
            top_5th_index = int(0.95 * n)

            top = s[:bottom_5th_index]  # Bottom 5% of the points
            bottom = s[top_5th_index:]
            
            # Perform PCA on i.rows
            i_pca, eigenvectors = PCA(i.rows)
            
            most_meaned = most - np.mean(i.rows, axis=0)  # Center `most` using mean of `i`
            done_meaned = done - np.mean(i.rows, axis=0)  # Center `done` using mean of `i`
            best_meaned = best - np.mean(i.rows, axis=0)
            rest_meaned = rest - np.mean(i.rows, axis=0)
            top_meaned = top - np.mean(i.rows, axis=0)
            bottom_meaned = bottom - np.mean(i.rows, axis=0)
            # Transform `most`, `done`, `best`, and `rest` using the PCA transformation

            most_pca = np.dot(most_meaned, eigenvectors)
            done_pca = np.dot(done_meaned, eigenvectors)
            best_pca = np.dot(best_meaned, eigenvectors)
            rest_pca = np.dot(rest_meaned, eigenvectors)
            top_pca = np.dot(top_meaned, eigenvectors)
            bottom_pca = np.dot(bottom_meaned, eigenvectors)

            # Select the correct subplot
            ax = axs[row_idx, col_idx]
            # Scatter plots
            ax.scatter(i_pca[:, 0], i_pca[:, 1], color='gray', label='i', alpha=0.5)
            ax.scatter(top_pca[:, 0], top_pca[:, 1], color = 'green', label = 'best', alpha = 0.5)
            ax.scatter(bottom_pca[:, 0], bottom_pca[:, 1], color = 'orange', label = 'rest', alpha = 0.5)
            ax.scatter(done_pca[:, 0], done_pca[:, 1], color='black', label='done', alpha=0.8)
            ax.scatter(most_pca[:, 0], most_pca[:, 1], color='yellow', marker = "v", label='most', alpha=1)
            ax.scatter(best_pca[:, 0], best_pca[:, 1], color='blue',marker = "v", label='initial best', alpha=1)
            ax.scatter(rest_pca[:, 0], rest_pca[:, 1], color='red', marker = "v", label='initial rest', alpha=1)
            # Labels and title
            if row_idx == 0:  # Only add title on the top row
                ax.set_title(f'{policy}', fontsize=12)
            if col_idx == 0:  # Only add y-axis label on the first column
                ax.set_ylabel('Principal Component 2')
            ax.set_xlabel('Principal Component 1')
            ax.legend(loc='upper right', fontsize='small')
    # Adjust layout to make room for labels and titles
    plt.tight_layout()
    # Save the figure
    os.makedirs('output/img/warms', exist_ok=True)
    dataset = dataset[dataset.rfind('/')+1:-4]
    plt.savefig(f'output/img/warms/{dataset}.png')
    # Show the plot
    plt.show() 







    

if __name__ == '__main__':
  # Command Line Visualizations for a quick sneek peak at the outputs
    if(len(sys.argv) ==3):
        visualize(dataset = sys.argv[1], show = sys.argv[2], save_fig=False, display = True)
    elif(len(sys.argv) == 2):
        visualize(dataset = sys.argv[1], save_fig=False, display = True)
    else:
        visualize(save_fig=False, display = True)
