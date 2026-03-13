import torch
from torchvision import datasets, transforms
import sys
import os

# 1. Setup Device
device = torch.device("cuda")

# 2. Re-declare the Architecture (Must match exactly)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(784, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 3. Load Test Data
transform = transforms.Compose([transforms.ToTensor()])
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)

# 4. Perform the Test
# Note: Since 'model' is in your current memory from the training script, 
# we can use it directly. If you closed the session, we'd load a saved file.
model.eval() 
correct = 0

print(f"Testing on 10,000 images using {torch.cuda.get_device_name(0)}...")

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # Get the index of the max log-probability (the predicted digit)
        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item()

accuracy = 100. * correct / len(test_loader.dataset)
print(f"\nResults:")
print(f"Total Correct: {correct}/10,000")
print(f"Accuracy: {accuracy}%")

print("Synchronizing GPU...")
if device.type == 'cuda':
    torch.cuda.synchronize()

print("Exiting...")
os._exit(0)