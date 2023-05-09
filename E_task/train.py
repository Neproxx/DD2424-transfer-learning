from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import OxfordPetsDataset, get_transforms
from model import OxfordPetsModel

DATASET_PATH = '/Users/mattiaevangelisti/Documents/OxfordPetsDataset'

def load_dataset(dataset):
    dataset = OxfordPetsDataset(root=DATASET_PATH, transform=get_transforms())
    train_size = int(0.75 * len(dataset))
    val_test_size = len(dataset) - train_size
    val_size = test_size = int(val_test_size / 2)
    train_dataset, val_test_dataset = random_split(dataset, [train_size, val_test_size])
    val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader

def test_model(model, dataloader, device):
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

    accuracy = 100 * running_corrects / len(dataloader.dataset)
    print(f'Test Accuracy: {accuracy:.4f}')

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

    test_model(model, test_loader, device)
    # torch.save(model.state_dict(), 'resnet18_binary.pth')
    # torch.save(model.state_dict(), 'resnet34_binary.pth')
    torch.save(model.state_dict(), 'resnet50_binary.pth')


if __name__ == '__main__':
    '''For NVIDIA GPU'''
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''For macOS'''
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = OxfordPetsModel(num_classes=2).to(device)

    print("The model will be running on", device, "device\n")
    train(model, num_epochs=10, dataset=OxfordPetsDataset, device=device)