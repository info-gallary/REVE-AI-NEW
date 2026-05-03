import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import DenseNet121_Weights
from torchvision import transforms, datasets, models
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os


transform_train = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = r"C:\Users\DELL\Downloads\archive"

# Load datasets
train_data = ImageFolder(root=f"{data_dir}/train", transform=transform_train)
val_data = ImageFolder(root=f"{data_dir}/test", transform=transform_test)
# train_data = datasets.ImageFolder('/kaggle/input/dermnet/train', transform=transform_train)
# test_data = datasets.ImageFolder('/kaggle/input/dermnet/test', transform=transform_test)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True,num_workers=4)
test_loader = DataLoader(val_data, batch_size=8, shuffle=False,num_workers=4)

if __name__ == "__main__":
# Move model to device
    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(train_data.classes))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using :",device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=100, save_path="models"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        best_acc = 0.0
        os.makedirs(save_path, exist_ok=True)
        
        for epoch in range(num_epochs):
            # Training Phase
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            
            # Validation Phase
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            
            val_acc = correct / total
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
            
            # Step scheduler
            scheduler.step(avg_loss)
            
            # Save Best Model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
            
        # Save Final Model
        torch.save(model.state_dict(), os.path.join(save_path, "final_model.pth"))
        print("✅ Training complete! Final and best models saved.")

    train_model(
        model=model, 
        criterion=criterion, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        train_loader=train_loader, 
        val_loader=test_loader,  
        num_epochs=100)
    