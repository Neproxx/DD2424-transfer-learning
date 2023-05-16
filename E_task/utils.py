import re
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    data = load_data('E_task/layer_accuracy.txt')
    plot_data(data)