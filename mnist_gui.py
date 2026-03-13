import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import ttk, filedialog
import threading
import time
import os
import numpy as np

# Define the same architecture as train_and_test.py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class MNISTGui:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognition Visualizer")
        self.root.geometry("650x750")
        self.root.configure(bg="#f5f5f5")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = Net().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
        self.running = False
        self.speed = 0.05
        
        self.setup_ui()
        self.load_data()
        
        # Check for existing model
        if os.path.exists("digits_model.pth"):
            try:
                self.model.load_state_dict(torch.load("digits_model.pth", map_location=self.device))
                self.info_var.set("Existing model loaded. Ready.")
            except:
                self.info_var.set("Ready to train (failed to load pth).")

    def setup_ui(self):
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        ttk.Label(self.main_frame, text="MNIST Visualizer", font=("Helvetica", 20, "bold")).pack(pady=10)
        
        # Visualization area (Top: Digit, Bottom: Bars)
        self.vis_frame = ttk.Frame(self.main_frame)
        self.vis_frame.pack(pady=10)
        
        # Digit Display
        self.canvas_digit = tk.Canvas(self.vis_frame, width=200, height=200, bg="black", highlightthickness=2, highlightbackground="#333")
        self.canvas_digit.grid(row=0, column=0, padx=20)
        
        # Confidence Bars
        self.canvas_bars = tk.Canvas(self.vis_frame, width=250, height=200, bg="white", highlightthickness=1, highlightbackground="#ccc")
        self.canvas_bars.grid(row=0, column=1, padx=20)
        self.bars = []
        self.bar_labels = []
        for i in range(10):
            x0 = 25 + i * 20
            y0 = 180
            x1 = 40 + i * 20
            y1 = 180
            bar = self.canvas_bars.create_rectangle(x0, y0, x1, y1, fill="#4a90e2")
            self.bars.append(bar)
            self.canvas_bars.create_text(x0+7, 190, text=str(i), font=("Helvetica", 8))

        # Prediction Result Label
        self.result_var = tk.StringVar(value="Prediction: -")
        self.result_label = ttk.Label(self.main_frame, textvariable=self.result_var, font=("Helvetica", 16, "bold"))
        self.result_label.pack(pady=10)
        
        # Info & Progress
        self.info_var = tk.StringVar(value="Select an action to begin")
        ttk.Label(self.main_frame, textvariable=self.info_var, font=("Helvetica", 10)).pack()
        
        self.progress = ttk.Progressbar(self.main_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=10)
        
        # Controls - Speed Slider
        speed_frame = ttk.Frame(self.main_frame)
        speed_frame.pack(pady=5)
        ttk.Label(speed_frame, text="Visualization Speed: ").pack(side=tk.LEFT)
        self.speed_slider = ttk.Scale(speed_frame, from_=0.2, to=0.0, orient=tk.HORIZONTAL, length=200, command=self.update_speed)
        self.speed_slider.set(0.05)
        self.speed_slider.pack(side=tk.LEFT)
        
        # Controls - Buttons
        self.btn_frame = ttk.Frame(self.main_frame)
        self.btn_frame.pack(pady=20)
        
        self.train_btn = ttk.Button(self.btn_frame, text="Start Live Training", command=self.start_training)
        self.train_btn.grid(row=0, column=0, padx=10)
        
        self.test_btn = ttk.Button(self.btn_frame, text="Test Random Image", command=self.test_random)
        self.test_btn.grid(row=0, column=1, padx=10)
        
        self.load_btn = ttk.Button(self.btn_frame, text="Load Custom Image", command=self.load_custom)
        self.load_btn.grid(row=0, column=2, padx=10)
        
        self.stop_btn = ttk.Button(self.main_frame, text="Stop / Reset", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(pady=5)

    def update_speed(self, val):
        self.speed = float(val)

    def load_data(self):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.train_set = datasets.MNIST('./data', train=True, download=True, transform=self.transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=1, shuffle=True)
        self.test_set = datasets.MNIST('./data', train=False, transform=self.transform)

    def update_visualization(self, tensor, target, output_probs, prediction):
        # 1. Update Digit Image
        img = transforms.ToPILImage()(tensor.squeeze())
        img = img.resize((200, 200), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas_digit.create_image(100, 100, image=self.photo)
        
        # 2. Update Confidence Bars
        for i in range(10):
            prob = output_probs[i].item()
            height = prob * 150
            y0 = 180 - height
            self.canvas_bars.coords(self.bars[i], 25 + i * 20, y0, 40 + i * 20, 180)
            
            if i == prediction:
                color = "#4caf50" if (target is None or i == target) else "#f44336"
                self.canvas_bars.itemconfig(self.bars[i], fill=color)
            else:
                self.canvas_bars.itemconfig(self.bars[i], fill="#4a90e2")

        # 3. Update Result Text
        target_str = str(target) if target is not None else "?"
        res_text = f"Target: {target_str} | Prediction: {prediction}"
        color = "green" if (target is None or target == prediction) else "red"
        self.result_label.configure(foreground=color)
        self.result_var.set(res_text)

    def training_loop(self):
        self.model.train()
        total_batches = len(self.train_loader)
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if not self.running:
                break
                
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            probs = torch.softmax(output, dim=1)[0].detach().cpu()
            prediction = output.argmax(dim=1, keepdim=True).item()
            
            self.root.after(0, self.update_visualization, data[0].cpu(), target.item(), probs, prediction)
            self.root.after(0, lambda b=batch_idx, l=loss.item(): self.update_status(b, total_batches, l))
            
            if self.speed > 0:
                time.sleep(self.speed)

        self.root.after(0, self.finish_training)

    def update_status(self, batch_idx, total, loss):
        self.progress['value'] = (batch_idx / total) * 100
        self.info_var.set(f"Training Batch {batch_idx}/{total} | Loss: {loss:.4f}")

    def start_training(self):
        self.running = True
        self.train_btn.configure(state=tk.DISABLED)
        self.test_btn.configure(state=tk.DISABLED)
        self.load_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        threading.Thread(target=self.training_loop, daemon=True).start()

    def test_random(self):
        self.model.eval()
        idx = torch.randint(0, len(self.test_set), (1,)).item()
        data, target = self.test_set[idx]
        data_in = data.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(data_in)
            probs = torch.softmax(output, dim=1)[0].cpu()
            prediction = output.argmax(dim=1, keepdim=True).item()
            
        self.update_visualization(data, target, probs, prediction)
        self.info_var.set(f"Tested random index {idx}")

    def load_custom(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if not file_path:
            return
            
        try:
            # 1. Convert to grayscale
            img = Image.open(file_path).convert('L')
            
            # 2. Extreme Auto-Contrast (Stretches darkest to 0 and brightest to 255)
            img = ImageOps.autocontrast(img, cutoff=2)
            
            # 3. Robust Inversion Detection
            # Instead of a simple mean, we look at the edges. 
            # If edges are bright, it's likely a black digit on white paper.
            data = np.array(img)
            edge_mean = (np.mean(data[0,:]) + np.mean(data[-1,:]) + np.mean(data[:,0]) + np.mean(data[:,-1])) / 4
            if edge_mean > 127:
                img = ImageOps.invert(img)
            
            # 4. Sharpen and Threshold
            # This aggressively pushes "grey" backgrounds to pure black (0)
            # and boosts "dim" digits toward white (255).
            img = img.point(lambda p: p * 1.5 if p > 80 else 0) 
            
            # Maintain aspect ratio:
            # 1. Resize so the largest dimension is 20 pixels (leaves room for padding like MNIST)
            img.thumbnail((20, 20), Image.Resampling.LANCZOS)
            
            # 2. Create a 28x28 black canvas
            new_img = Image.new('L', (28, 28), (0))
            
            # 3. Center the resized digit on the canvas
            offset = ((28 - img.width) // 2, (28 - img.height) // 2)
            new_img.paste(img, offset)
            img = new_img
            
            # Convert to tensor
            tensor = transforms.ToTensor()(img)
            data_in = tensor.unsqueeze(0).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                output = self.model(data_in)
                probs = torch.softmax(output, dim=1)[0].cpu()
                prediction = output.argmax(dim=1, keepdim=True).item()
                
            self.update_visualization(tensor, None, probs, prediction)
            self.info_var.set(f"Loaded custom image: {os.path.basename(file_path)}")
            
        except Exception as e:
            self.info_var.set(f"Error loading image: {str(e)}")

    def stop(self):
        self.running = False
        self.train_btn.configure(state=tk.NORMAL)
        self.test_btn.configure(state=tk.NORMAL)
        self.load_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)

    def finish_training(self):
        self.stop()
        torch.save(self.model.state_dict(), "digits_model.pth")
        self.info_var.set("Training finished and model saved to 'digits_model.pth'")

if __name__ == "__main__":
    root = tk.Tk()
    app = MNISTGui(root)
    root.mainloop()
