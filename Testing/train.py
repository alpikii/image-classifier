import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader  # Added
from resnet18 import ResNet18
import torchvision.transforms as transforms, torchvision, matplotlib.pyplot as plt
import matplotlib.pyplot as plt  # Added
import numpy as np  # Added
from collections import Counter

model = ResNet18()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Added
model.to(device)  # Added

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

class NumpyDataset(Dataset):

    def __init__(self, image_path, label_path, transform=None):
        self.images = np.load(image_path)
        self.labels = np.load(label_path)
        self.transform = transform
        # Convert image format from (N, H, W, C) to (N, C, H, W)
        self.images = self.images.transpose((0, 3, 1, 2)).astype(np.float32)
        self.labels = self.labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

print("Downloading dataset")
transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

trainset = NumpyDataset('./cifar10/cifar10_train_images_cat_10.npy',
                         './cifar10/cifar10_train_labels_cat_10.npy',
                         transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
print(Counter(trainset.labels))

print("Starting training...")
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Added by Salla
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch} loss: {running_loss / len(trainloader)}')

print("Training done")
print("Downloading test dataset")
testset = NumpyDataset('./cifar/cifar10_test_images.npy',
                       './cifar/cifar10_test_labels.npy',
                       transform=transform)
print(Counter(testset.labels))
# testset = torchvision.datasets.CIFAR10(root='./data',
#                                       train=False,
#                                       download=True,
#                                       transform=transforms.Compose([
#                                           transforms.ToTensor(),
#                                           transforms.Normalize((0.5, 0.5, 0.5),
#                                                                (0.5, 0.5, 0.5))]))
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

print("Testing accuracy...")
correct = 0
total = 0
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for data in testloader:  # Or use testloader if you have a separate test set
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')
