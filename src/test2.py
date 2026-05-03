import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
from model import SkinDiseaseCNN  # First model

def load_first_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkinDiseaseCNN(num_classes=9).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device

def load_second_model(weights_path, num_classes):
    model = models.densenet121(weights=None)
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def transform_image(image_path, resize):
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict(image_path, model, device, class_names, resize):
    image = transform_image(image_path, resize).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    return class_names[predicted_class], confidence

def main(image_path):
    # Class labels
    class_names_1 = [
        "Actinic keratosis", "Atopic Dermatitis", "Benign keratosis",
        "Dermatofibroma", "Melanocytic nevus", "Melanoma", "Squamous cell carcinoma",
        "Tinea Ringworm Candidiasis", "Vascular lesion"
    ]
    
    class_names_2 = [
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
    
    # Load models
    model1, device1 = load_first_model(r"C:\Users\DELL\OneDrive\CodeDB\Project\PY_ML_DL_GenAI\Skin_Cancer\src\models\skin_disease_model.pth")
    model2, device2 = load_second_model(r"C:\Users\DELL\Downloads\model_epoch_25.pth", num_classes=23)
    
    # Get predictions
    pred1, conf1 = predict(image_path, model1, device1, class_names_1, (128, 128))
    pred2, conf2 = predict(image_path, model2, device2, class_names_2, (512, 512))
    
    # Print the highest confidence prediction
    print(f"Prediction: {pred1} ({conf1*100:.2f}% confidence), {pred2} ({conf2*100:.2f}% confidence)" if conf1 > conf2 else f"Prediction: {pred2} ({conf2*100:.2f}% confidence), {pred1} ({conf1*100:.2f}% confidence)")


# Test with an image
test_images = [
# r"C:\Users\DELL\Downloads\data1\train\Melanoma\ISIC_0000297.jpg",
# r"C:\Users\DELL\Downloads\data1\train\Benign keratosis\ISIC_0014625_downsampled.jpg",
# r"C:\Users\DELL\Downloads\skin samples\samples\j2.jpg",
# r"C:\Users\DELL\Downloads\data1\train\Melanoma\ISIC_0000294.jpg",
# r"C:\Users\DELL\Downloads\skin samples\samples\r1.jpg",
# r"C:\Users\DELL\Downloads\skin samples\samples\j3.jpg",
# r"C:\Users\DELL\Downloads\data1\val\Tinea Ringworm Candidiasis\aug_0_pha-259018.jpg",
# r"C:\Users\DELL\Downloads\skin samples\samples\n1.jpg"
# r"C:\Users\DELL\Downloads\archive\test\Acne and Rosacea Photos\rosacea-116.jpg",
#    r"C:\Users\DELL\Downloads\data1\train\Benign keratosis\ISIC_0014625_downsampled.jpg",
#    r"C:\Users\DELL\Downloads\archive\test\Light Diseases and Disorders of Pigmentation\sun-damaged-skin-35.jpg",
#     r"C:\Users\DELL\Downloads\archive\test\Poison Ivy Photos and other Contact Dermatitis\metal-dermatitis-9.jpg"
# r"C:\Users\DELL\Downloads\archive\train\Acne and Rosacea Photos\acne-pustular-7.jpg"
r"C:\Users\DELL\Downloads\skin samples\samples\j3.jpg"
]

for img_path in test_images:
    main(img_path)
