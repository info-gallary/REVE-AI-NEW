import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn as nn
from model import SkinDiseaseCNN

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

data_dir = r"C:\Users\DELL\Downloads\data1"

# Load datasets
train_dataset = ImageFolder(root=f"{data_dir}/train", transform=transform)
val_dataset = ImageFolder(root=f"{data_dir}/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = 32  # Update to match the number of skin disease classes
    model = SkinDiseaseCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Training Accuracy Calculation
            _, predicted_train = torch.max(outputs, 1)
            correct_train += (predicted_train == labels).sum().item()
            total_train += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train

        # Validation Accuracy Calculation
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted_val = torch.max(outputs, 1)
                correct_val += (predicted_val == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Training Accuracy: {train_acc*100:.2f}%, Validation Accuracy: {val_acc*100:.2f}%")

    # Save Model
    torch.save(model.state_dict(), "skin_disease_model_1000.pth")
    print("✅ Training complete! Model saved.")
