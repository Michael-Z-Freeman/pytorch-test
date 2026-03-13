import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import sys
import os

# 1. Setup Device - The bundle maps AMD to 'cuda' for compatibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {torch.cuda.get_device_name(0)}")

# 2. Simple Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 3. Data & Training Setup
transform = transforms.Compose([transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 4. Training Loop (Just 1 Epoch for speed)
model.train()
print("Starting training...")
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % 100 == 0:
        print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

print("Training Complete! Synchronizing GPU...")
if device.type == 'cuda':
    torch.cuda.synchronize()

print("Exiting...")
os._exit(0)