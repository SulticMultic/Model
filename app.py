import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# Load the model
model = load_model('GTSRB_model.h5')

# Function to make predictions (for image input)
def predict_image(image):
    # Preprocess the image (resize to 224x224, convert to array, normalize, etc.)
    image = image.resize((224, 224))  # Resize image to 224x224
    image = np.array(image)           # Convert image to numpy array
    image = image.astype('float32') / 255.0  # Normalize to 0-1 range
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 224, 224, 3)
    
    # Prediction
    prediction = model.predict(image)
    return prediction

# Streamlit UI
st.title("Image Prediction with Model")
st.write("Upload an image to make a prediction.")

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the image
    image = Image.open(uploaded_image)
    
    # Make a prediction
    if st.button("Make Prediction"):
        result = predict_image(image)
        st.write(f"Prediction: {result}")
