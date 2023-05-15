import itertools
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset_binary import OxfordPetsDatasetBinary ,get_transforms
from dataset_multi import OxfordPetsDatasetMulti
from model import OxfordPetsModel
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH = '/Users/mattiaevangelisti/Documents/OxfordPetsDataset'

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def load_dataset(dataset):
    train_size = int(0.75 * len(dataset))
    val_test_size = len(dataset) - train_size
    val_size = test_size = int(val_test_size / 2)
    train_dataset, val_test_dataset = random_split(dataset, [train_size, val_test_size])
    val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader

def test_model(model, dataloader, device, classes_name):
    model.eval()
    running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            running_corrects += torch.sum(preds == labels.data)

    accuracy = 100 * running_corrects / len(dataloader.dataset)
    print(f'Test Accuracy: {accuracy:.4f}')

    cm = confusion_matrix(all_labels, all_preds)
    np.set_printoptions(precision=2)

    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(cm, classes=classes_name, title='Confusion matrix')

    plt.show()

def train(model, num_epochs, dataset, device):
    train_loader, val_loader, test_loader = load_dataset(dataset)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

    test_model(model, test_loader, device, dataset.classes)
    # torch.save(model.state_dict(), 'resnet18_binary.pth')
    # torch.save(model.state_dict(), 'resnet34_binary.pth')
    # torch.save(model.state_dict(), 'resnet50_binary.pth')
    torch.save(model.state_dict(), 'E_task/resnet18_multiclass.pth')


if __name__ == '__main__':
    '''For NVIDIA GPU'''
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''For macOS'''
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("The model will be running on", device, "device\n")

    # for binary classification
    # dataset = OxfordPetsDatasetBinary(root=DATASET_PATH, transform=get_transforms())
    # for multiclass classification
    dataset = OxfordPetsDatasetMulti(root=DATASET_PATH, transform=get_transforms())

    model = OxfordPetsModel(num_classes=dataset.classes).to(device)
    train(model, num_epochs=10, dataset=dataset, device=device)