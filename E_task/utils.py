import re
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

def load_data(file_name):
    # Define regular expressions to match the dictionary format
    key_pattern = re.compile(r"(\d+): tensor")
    value_pattern = re.compile(r"tensor\(([\d\.]+)")

    with open(file_name, 'r') as file:
        content = file.read()

    # Find all keys and values
    keys = map(int, key_pattern.findall(content))
    values = map(float, value_pattern.findall(content))

    # Combine keys and values into a dictionary
    data_dict = dict(zip(keys, values))

    return data_dict


def plot_data(data):
    # create lists of layers and accuracies from the dictionary
    layers = list(data.keys())
    accuracies = list(data.values())

    fig, ax = plt.subplots()
    ax.plot(layers, accuracies, marker='o', color='orange')
    ax.set_xlabel('Number of Layers Finetuned')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs. Number of Layers Finetuned')
    ax.grid(True)
    
    # Set the background color of the plot area to gray
    ax.set_facecolor('gainsboro')

    plt.show()

def parse_accuracy_data(filename):
    with open(filename, 'r') as f:
        raw_data = f.read()

    # Remove the 'tensor' and 'device' parts from the data
    cleaned_data = re.sub(r'tensor\(([\d\.]+), device=\'mps:0\'\)', r'\1', raw_data)

    # Parse the cleaned data into a dictionary
    data_dict = ast.literal_eval(cleaned_data)
    
    layer_accuracies = {}

    # Iterate over each seed in the dictionary
    for seed, accuracies in data_dict.items():
        # Iterate over each layer in the accuracies dictionary
        for layer, accuracy in accuracies.items():
            # If this layer is not already in the layer_accuracies dictionary, add it
            if layer not in layer_accuracies:
                layer_accuracies[layer] = []
            # Add the accuracy for this seed and layer to the layer_accuracies dictionary
            layer_accuracies[layer].append(float(accuracy))

    # Compute the average accuracy for each layer
    avg_layer_accuracies = {layer: np.mean(accuracies) for layer, accuracies in layer_accuracies.items()}

    return avg_layer_accuracies

def plot_heatmap(filename):
    with open(filename, 'r') as f:
        raw_data = f.read()

    # Remove the 'tensor' and 'device' parts from the data
    cleaned_data = re.sub(r'tensor\(([\d\.]+), device=\'mps:0\'\)', r'\1', raw_data)

    # Parse the cleaned data into a dictionary
    data_dict = ast.literal_eval(cleaned_data)

    # Initialize empty dataframe
    df = pd.DataFrame(columns=['Learning Rate', 'Scheduler', 'Accuracy'])

    # Fill dataframe with data
    for k, v in data_dict.items():
        lr, sched = k
        acc = v # Convert tensor to a Python number
        # add data to dataframe
        df = pd.concat([df, pd.DataFrame([[lr, sched, acc]], columns=['Learning Rate', 'Scheduler', 'Accuracy'])], ignore_index=True)

    # Create a pivot table from the dataframe
    pivot = df.pivot(index='Learning Rate', columns='Scheduler', values='Accuracy')
    # replace NaN column name with 'None' string
    pivot.columns = pivot.columns.fillna('None')
    # plot heatmap with two decimal digits for accuracy
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': 'Accuracy'})
    plt.show()
    
def show_images(dataset, num_images=20):
    fig = plt.figure(figsize=(20, 4))
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    # Randomly select num_images indices from the dataset
    indices = np.random.choice(len(dataset), num_images, replace=False)
    
    for i, idx in enumerate(indices):
        input, _ = dataset[idx]
        ax = fig.add_subplot(2, num_images//2, i + 1, xticks=[], yticks=[])
        input = input * std[..., None, None] + mean[..., None, None]  # denormalize
        input = input.permute(1, 2, 0)  # switch from C,H,W to H,W,C
        input = torch.clamp(input, 0, 1)
        ax.imshow(input.numpy())
    plt.show()




if __name__ == "__main__":
    data = load_data('E_taskresults//layer_accuracy.txt')
    plot_data(data)
    # print best accuracy and corresponding number of layers
    print(max(data.values()), max(data, key=data.get))
    average = parse_accuracy_data('E_task/results/first_10_layer_accuracy.txt')
    # dispòay the average accuracy for each layer as a bar chart
    plot_data(average)
    plot_heatmap('E_task/results/grid_search.txt')