import argparse
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
import os

def get_data_loaders(data_dir, batch_size=64):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, train_dataset.class_to_idx

def build_model(arch='vgg16', hidden_units=512, gpu=True):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = model.classifier[0].in_features
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        in_features = model.classifier[1].in_features
    else:
        raise ValueError('Invalid architecture choice. Choose either vgg16 or alexnet.')

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    if gpu and torch.cuda.is_available():
        model = model.cuda()

    return model

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=5, save_path='checkpoint.pth', arch='vgg16'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = (correct / total) * 100
        print(f"Validation Loss: {test_loss/len(test_loader):.4f}")
        print(f"Validation Accuracy: {accuracy:.2f}%")

    # Save the model checkpoint
    torch.save({
        'arch': arch,
        'state_dict': model.state_dict(),
        'class_to_idx': train_loader.dataset.class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs
    }, save_path)
    print(f"Checkpoint saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset of images.')
    parser.add_argument('data_dir', type=str, help='Directory containing the dataset.')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save the checkpoint.')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'alexnet'], help='Model architecture.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training.')

    args = parser.parse_args()

    train_loader, test_loader, class_to_idx = get_data_loaders(args.data_dir)
    model = build_model(arch=args.arch, hidden_units=args.hidden_units, gpu=args.gpu)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    train_model(model, train_loader, test_loader, criterion, optimizer, epochs=args.epochs, save_path=args.save_dir, arch=args.arch)
