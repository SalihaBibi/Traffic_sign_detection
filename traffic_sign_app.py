import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Define the transformation for preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load the model
model = models.resnet18()  # Initialize model architecture
model.fc = torch.nn.Linear(model.fc.in_features, 43)  # Update for 43 classes
model.load_state_dict(torch.load("traffic_sign_model.pth"))
model.eval()

# Define class labels
class_labels = {0: "Speed Limit 20", 1: "Speed Limit 30", ..., 42: "End of No Passing for Vehicles over 3.5T"}

# Streamlit app
st.title("Traffic Sign Detection")
st.write("Upload an image to classify the traffic sign!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)  # Transform and add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_class_index = torch.argmax(outputs, 1).item()  # Get index
        predicted_class_label = class_labels[predicted_class_index]  # Map to label

    st.write(f"Predicted Traffic Sign: {predicted_class_label}")
