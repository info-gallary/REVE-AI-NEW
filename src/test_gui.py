import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import SkinDiseaseCNN
import os

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkinDiseaseCNN().to(device)
model.load_state_dict(torch.load("skin_disease_model.pth"))
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Streamlit app
def predict(image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    output = model(image).item()
    image_name = "Uploaded Image"
    result = 'Diseased' if output > 0.5 else 'Healthy'
    confidence = output * 100
    return result, confidence, image_name

# Streamlit UI
st.title("Skin Disease Prediction")
st.write("Upload an image to predict whether the skin is Diseased or Healthy")

# Image upload widget
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_image is not None:
    # Open the uploaded image
    image = Image.open(uploaded_image)
    
    # Create two columns: one for the result and one for the image
    col1, col2 = st.columns([1, 3])
    
    # Left column: Display the result
    with col1:
        result, confidence, image_name = predict(image)
        st.write(f"Prediction: {result} ({confidence:.2f}% confidence)")
        st.write(f"Image: {image_name}")
    
    # Right column: Display the image
    with col2:
        st.image(image, caption="Uploaded Image", use_column_width=True)
