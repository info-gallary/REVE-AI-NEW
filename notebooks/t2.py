import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import os

# Fix OpenMP error
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import our improved model
from model1 import SkinDiseaseCNN

# Create directory for saving graphs
os.makedirs('graphs', exist_ok=True)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = r"C:\Users\DELL\Downloads\data1"

# Load datasets
train_dataset = ImageFolder(root=f"{data_dir}/train", transform=transform)
val_dataset = ImageFolder(root=f"{data_dir}/val", transform=transform)

# Print dataset information
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
num_classes = len(train_dataset.classes)
model = SkinDiseaseCNN(num_classes=num_classes).to(device)
print(f"Model initialized with {num_classes} output classes")

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)

# Lists to store metrics for plotting
train_losses = []
val_losses = []
train_accs = []
val_accs = []
epochs = []

# Training Loop
num_epochs = 100
best_val_acc = 0.0

print("Starting training...")
for epoch in range(num_epochs):
    start_time = time.time()
    
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss += loss.item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    
    # Store metrics for plotting
    epochs.append(epoch + 1)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    # Print metrics
    elapsed_time = time.time() - start_time
    print(f"Epoch [{epoch+1}/{num_epochs}], Time: {elapsed_time:.2f}s")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
    
    # Track best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "skin_disease_model_best.pth")
        print(f"New best model with validation accuracy: {val_acc*100:.2f}%")

# Save final model
torch.save(model.state_dict(), "skin_disease_model.pth")
print("✅ Training complete! Model saved.")
print(f"Best validation accuracy achieved: {best_val_acc*100:.2f}%")

# Plot final graphs
print("Generating final training graphs...")

# Plot learning curves
plt.figure(figsize=(15, 10))

# Plot accuracy
plt.subplot(2, 1, 1)
plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

# Plot loss
plt.subplot(2, 1, 2)
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.savefig('graphs/training_curves.png', dpi=300)

# Create focused subplots for better visualization of trends
plt.figure(figsize=(20, 12))

# Full training period
plt.subplot(2, 2, 1)
plt.plot(epochs, train_accs, 'b-', label='Training')
plt.plot(epochs, val_accs, 'r-', label='Validation')
plt.title('Accuracy - Full Training', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(epochs, train_losses, 'b-', label='Training')
plt.plot(epochs, val_losses, 'r-', label='Validation')
plt.title('Loss - Full Training', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

# Last 100 epochs (or fewer if training for less)
last_n = min(100, len(epochs))
plt.subplot(2, 2, 3)
plt.plot(epochs[-last_n:], train_accs[-last_n:], 'b-', label='Training')
plt.plot(epochs[-last_n:], val_accs[-last_n:], 'r-', label='Validation')
plt.title(f'Accuracy - Last {last_n} Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(epochs[-last_n:], train_losses[-last_n:], 'b-', label='Training')
plt.plot(epochs[-last_n:], val_losses[-last_n:], 'r-', label='Validation')
plt.title(f'Loss - Last {last_n} Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.savefig('graphs/training_analysis.png', dpi=300)
print("Final graphs saved as 'training_curves.png' and 'training_analysis.png' in the 'graphs' folder.")