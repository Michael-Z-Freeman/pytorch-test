import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. Setup Device
device = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# 2. Define Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 3. Load Data
transform = transforms.Compose([transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 4. Train (1 Epoch)
model.train()
print("Training for 1 epoch...")
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# 5. Test (Immediately following training)
model.eval()
correct = 0
print("Testing on 10,000 images...")
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

print(f"\nSuccess! Final Accuracy: {100. * correct / len(test_loader.dataset)}%")

# 6. Save the 'Brain' so we don't lose it again
torch.save(model.state_dict(), "digits_model.pth")
print("Model saved to 'digits_model.pth'")