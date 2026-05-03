import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import VGG19
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def predict_skin_disease(image_path):
    # Define class names
    class_names = ['Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases',
                   'Acne and Rosacea Photos', 'Systemic Disease', 'Poison Ivy Photos and other Contact Dermatitis',
                   'Vascular Tumors', 'Urticaria Hives', 'Atopic Dermatitis Photos', 'Bullous Disease Photos',
                   'Hair Loss Photos Alopecia and other Hair Diseases', 'Tinea Ringworm Candidiasis and other Fungal Infections',
                   'Psoriasis pictures Lichen Planus and related diseases', 'Melanoma Skin Cancer Nevi and Moles',
                   'Nail Fungus and other Nail Disease', 'Scabies Lyme Disease and other Infestations and Bites',
                   'Eczema Photos', 'Exanthems and Drug Eruptions', 'Herpes HPV and other STDs Photos',
                   'Seborrheic Keratoses and other Benign Tumors',
                   'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
                   'Vasculitis Photos', 'Cellulitis Impetigo and other Bacterial Infections',
                   'Warts Molluscum and other Viral Infections']

    # Load VGG19 model for feature extraction
    vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(180, 180, 3))

    # Load trained model (which expects VGG19 features)
    model = tf.keras.models.load_model(r'c:\Users\DELL\Downloads\pred.h5', compile=False)

    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (180, 180))
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Extract features using VGG19
    img_features = vgg_model.predict(img)  # Output shape: (1, 5, 5, 512)
    img_features = img_features.reshape(1, -1)  # Flatten to (1, 12800) or similar

    # Predict
    pred = model.predict(img_features)[0]
    predicted_class_index = np.argmax(pred)
    predicted_class_name = class_names[predicted_class_index]
    confidence_score = pred[predicted_class_index]

    print(f"Predicted Class: {predicted_class_name}, Confidence Score: {confidence_score:.2f}")

# Test
predict_skin_disease(r"C:\Users\DELL\Downloads\archive\test\Urticaria Hives\cholinergic-uriticaria-12.jpg")
