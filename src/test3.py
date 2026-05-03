import torch
from torchvision import models, transforms
from PIL import Image
import os

# Define the same model architecture
def load_model(weights_path, num_classes):
    model = models.densenet121(weights=None)  # No pretrained weights
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(num_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, num_classes)
    )
    
    # Load trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Define the same transformations as during training/testing
transform_test = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Prediction function
def predict(image_path, model, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path).convert("RGB")
    image = transform_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)  # Get probabilities
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    image_name = os.path.basename(image_path)
    print(f"Predicted: {class_names[predicted_class]} ({confidence*100:.2f}% confidence) | Image: {image_name}")

# Load the trained model
weights_path = r"C:\Users\DELL\Downloads\model_epoch_25.pth"  # Change to "final_model.pth" if needed
num_classes = 23  # Adjust based on your dataset
class_names = [
    "Acne and Rosacea Photos", "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis Photos", "Bullous Disease Photos", "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema Photos", "Exanthems and Drug Eruptions", "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs Photos", "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases", "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease", "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases", "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses and other Benign Tumors", "Systemic Disease", "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives", "Vascular Tumors", "Vasculitis Photos", "Warts Molluscum and other Viral Infections"
]

# Load Model
model = load_model(weights_path, num_classes)

# Test Images
test_images = [
    # r"C:\Users\DELL\Downloads\data1\train\Melanoma\ISIC_0000297.jpg",
    # r"C:\Users\DELL\Downloads\data1\train\Benign keratosis\ISIC_0014625_downsampled.jpg",
    # r"C:\Users\DELL\Downloads\archive\test\Acne and Rosacea Photos\perioral-dermatitis-100.jpg",
    # r"C:\Users\DELL\Downloads\archive\test\Acne and Rosacea Photos\rosacea-116.jpg",
    # r"C:\Users\DELL\Downloads\archive\test\Light Diseases and Disorders of Pigmentation\sun-damaged-skin-35.jpg",
    #   r"C:\Users\DELL\Downloads\data1\train\Benign keratosis\ISIC_0014625_downsampled.jpg",
   r"C:\Users\DELL\Downloads\archive\test\Acne and Rosacea Photos\rosacea-116.jpg",
#    r"C:\Users\DELL\Downloads\archive\test\Poison Ivy Photos and other Contact Dermatitis\metal-dermatitis-9.jpg",
    # r"C:\Users\DELL\Downloads\archive\test\Urticaria Hives\cholinergic-uriticaria-12.jpg"
    r"C:\Users\DELL\Downloads\archive\train\Acne and Rosacea Photos\acne-pustular-7.jpg",
    r"C:\Users\DELL\Downloads\archive\train\Eczema Photos\stasis-dermatitis-81.jpg",
    r"C:\Users\DELL\Downloads\archive\train\Light Diseases and Disorders of Pigmentation\vitiligo-42.jpg",
    r"C:\Users\DELL\Downloads\archive\train\Acne and Rosacea Photos\acne-pustular-7.jpg"

]

# Run Predictions
for img_path in test_images:
    predict(img_path, model, class_names)
