import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO
import os

device = torch.device("cuda")
print(f"Deep Learning on: {torch.cuda.get_device_name(0)}")

# 1. Image Loader
def load_image(path, max_size=400, shape=None):
    img = Image.open(path).convert('RGB')
    
    if shape:
        size = shape
    else:
        size = min(max(img.size), max_size)
        
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return in_transform(img).unsqueeze(0).to(device)

# Load images: first the content, then style resized to match content
content = load_image("Cat_November_2010-1a.jpg")
style = load_image("Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg", shape=content.shape[-2:])

# 2. VGG19 Model (The 'Art Critic')
vgg = models.vgg19(weights='DEFAULT').features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad_(False)

# 3. Create the 'Canvas' (Target Image)
target = content.clone().requires_grad_(True).to(device)
optimizer = optim.Adam([target], lr=0.03)

print("Starting Neural Style Transfer (300 iterations)...")
for i in range(1, 301):
    optimizer.zero_grad()
    # Simple Content Loss: keep the cat shape
    content_loss = torch.mean((target - content)**2)
    # Simple Style Loss: pull in the Van Gogh colors/swirls
    style_loss = torch.mean((target - style)**2)
    
    total_loss = content_loss + (style_loss * 10)
    total_loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(f"Iteration {i}/300 | Loss: {total_loss.item():.4f}")

# 4. Save result
image = target.to("cpu").clone().detach().squeeze(0)
image = transforms.ToPILImage()(image)
image.save("output_art.png")
print("Complete! Result saved as 'output_art.png'")

print("Synchronizing GPU...")
if device.type == 'cuda':
    torch.cuda.synchronize()

print("Exiting...")
os._exit(0)