import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from resnet18 import ResNet18
import torchvision.transforms as transforms, torchvision, matplotlib.pyplot as plt

model = ResNet18()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print("Downloading dataset")
trainset = torchvision.datasets.CIFAR10(root='./data', 
                                        train=True, 
                                        download=True,
                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=4, 
                                          shuffle=True)

print("Starting training...")
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch} loss: {running_loss / len(trainloader)}')
  
print("Training done")
print("Downloading test dataset")
testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5),
                                                                (0.5, 0.5, 0.5))]))
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

print("Testing accuracy...")
correct = 0
total = 0
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy on training set: {100 * correct / total:.2f}%')
