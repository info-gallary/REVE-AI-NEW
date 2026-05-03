import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import shutil

# Define transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Custom Dataset
class UnlabeledDataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.img_folder = img_folder
        self.transform = transform
        self.img_list = os.listdir(img_folder)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_folder, self.img_list[idx])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.img_list[idx]

# Function to process and label images
def label_images(input_folder, output_folder):
    dataset = UnlabeledDataset(input_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  
    model = model.to(device)
    model.eval()

    features, filenames = [], []
    with torch.no_grad():
        for images, names in dataloader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(outputs.size(0), -1).cpu().numpy()
            features.append(outputs)
            filenames.extend(names)

    features = np.vstack(features)

    # K-Means Clustering (2 classes: healthy & diseased)
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(features)

    # Create labeled directories
    healthy_folder = os.path.join(output_folder, "healthy")
    disease_folder = os.path.join(output_folder, "disease")
    os.makedirs(healthy_folder, exist_ok=True)
    os.makedirs(disease_folder, exist_ok=True)

    # Move images
    for i, filename in enumerate(filenames):
        src_path = os.path.join(input_folder, filename)
        dest_folder = healthy_folder if clusters[i] == 0 else disease_folder
        shutil.move(src_path, os.path.join(dest_folder, filename))

    print(f"✅ Auto-labeling complete! Images moved to '{output_folder}'.")

# Process train and validation datasets
data = r"C:\Users\DELL\Downloads\data"
label_images(f"{data}/train", f"{data}/train_labeled")
label_images(f"{data}/validation", f"{data}/validation_labeled")
