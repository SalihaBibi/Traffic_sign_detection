import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import os

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model if file exists
model_path = "traffic_sign_model_resnet18.pth"
if os.path.exists(model_path):
    # Define and load the model
    model = models.resnet18(weights=None)  # No pre-trained weights
    model.fc = nn.Linear(model.fc.in_features, 43)  # Adjust final layer for 43 classes
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load model weights
    model = model.to(device)
    model.eval()
else:
    st.error(f"Model file '{model_path}' not found. Please ensure the file is in the correct directory.")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Streamlit UI for uploading an image and making a prediction
st.title("Traffic Sign Recognition")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Only run prediction if model loaded successfully
    if 'model' in locals():
        # Preprocess the image
        image = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            st.write(f"Predicted Class: {predicted.item()}")
