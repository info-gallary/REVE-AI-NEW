import torch
import torchvision.transforms as transforms
from PIL import Image
from model import SkinDiseaseCNN
# from model1 import SkinDiseaseCNN
import os

# Define class labels (based on your dataset structure)
CLASS_NAMES = [
    "Actinic keratosis",
    "Atopic Dermatitis",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanocytic nevus",
    "Melanoma",
    "Squamous cell carcinoma",
    "Tinea Ringworm Candidiasis",
    "Vascular lesion"
]

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkinDiseaseCNN(num_classes=9).to(device)  # Ensure correct num_classes
model.load_state_dict(torch.load(r"C:\Users\DELL\OneDrive\CodeDB\Project\PY_ML_DL_GenAI\Skin_Cancer\src\models\skin_disease_model.pth", map_location=device))
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load and Predict
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)  # Get probabilities
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Pred.: {CLASS_NAMES[predicted_class]} ({confidence*100:.2f}% confidence) | Image: {image_name}")

# Test on new images
test_images = [
# r"C:\Users\DELL\Downloads\data1\train\Melanoma\ISIC_0000297.jpg",
# r"C:\Users\DELL\Downloads\data1\train\Benign keratosis\ISIC_0014625_downsampled.jpg",
# r"C:\Users\DELL\Downloads\skin samples\samples\j2.jpg",
# r"C:\Users\DELL\Downloads\data1\train\Melanoma\ISIC_0000294.jpg",
# r"C:\Users\DELL\Downloads\skin samples\samples\r1.jpg",
# r"C:\Users\DELL\Downloads\skin samples\samples\j3.jpg",
# r"C:\Users\DELL\Downloads\data1\val\Tinea Ringworm Candidiasis\aug_0_pha-259018.jpg",
# r"C:\Users\DELL\Downloads\skin samples\samples\n1.jpg"
r"C:\Users\DELL\Downloads\data1\train\Squamous cell carcinoma\ISIC_0027184.jpg",
r"C:\Users\DELL\Downloads\WhatsApp Image 2025-03-29 at 18.06.49_8a657774.jpg"
]

for img_path in test_images:
    predict(img_path)
