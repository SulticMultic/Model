import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model = load_model('enhanced_model.h5')

# Function to make predictions
def predict(input_data):
    # Example prediction (you can modify based on your model's input)
    prediction = model.predict(np.array([input_data]))
    return prediction

# Streamlit UI
st.title("Streamlit Model Testing")
input_data = st.number_input("Enter input data", min_value=0, max_value=100)
if st.button("Make Prediction"):
    result = predict(input_data)
    st.write(f"Prediction: {result}")
