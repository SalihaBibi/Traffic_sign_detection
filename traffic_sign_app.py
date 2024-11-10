import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchvision import models
import torch.nn as nn

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 43)  # Modify for GTSRB classes
model.load_state_dict(torch.load("traffic_sign_model_resnet18.pth"))
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Streamlit UI for uploading image and making prediction
st.title("Traffic Sign Recognition")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model to device
model = model.to(device)


if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    
    # Make prediction
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    st.image(image.cpu().squeeze(), caption="Uploaded Image", use_column_width=True)
    st.write(f"Predicted Class: {predicted.item()}")