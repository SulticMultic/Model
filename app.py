import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the model
model = load_model('enhanced_model.h5')

# Define the class labels as per your dictionary
classes = {  
    1:'Speed limit (20km/h)', 2:'Speed limit (30km/h)', 3:'Speed limit (50km/h)', 
    4:'Speed limit (60km/h)', 5:'Speed limit (70km/h)', 6:'Speed limit (80km/h)', 
    7:'End of speed limit (80km/h)', 8:'Speed limit (100km/h)', 9:'Speed limit (120km/h)', 
    10:'No passing', 11:'No passing veh over 3.5 tons', 12:'Right-of-way at intersection', 
    13:'Priority road', 14:'Yield', 15:'Stop', 16:'No vehicles', 
    17:'Veh > 3.5 tons prohibited', 18:'No entry', 19:'General caution', 
    20:'Dangerous curve left', 21:'Dangerous curve right', 22:'Double curve', 
    23:'Bumpy road', 24:'Slippery road', 25:'Road narrows on the right', 
    26:'Road work', 27:'Traffic signals', 28:'Pedestrians', 
    29:'Children crossing', 30:'Bicycles crossing', 31:'Beware of ice/snow', 
    32:'Wild animals crossing', 33:'End speed + passing limits', 
    34:'Turn right ahead', 35:'Turn left ahead', 36:'Ahead only', 
    37:'Go straight or right', 38:'Go straight or left', 39:'Keep right', 
    40:'Keep left', 41:'Roundabout mandatory', 42:'End of no passing', 
    43:'End no passing veh > 3.5 tons'
}

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
st.title("Traffic Sign Prediction")
st.write("Upload an image of a traffic sign to make a prediction.")

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the image
    image = Image.open(uploaded_image)
    
    # Make a prediction
    if st.button("Make Prediction"):
        result = predict_image(image)
        
        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(result)
        
        # Get the corresponding class label from the classes dictionary
        predicted_class = classes[predicted_class_index + 1]  # Adding 1 to match the index
        
        # Display the prediction result
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Class probabilities: {result}")
