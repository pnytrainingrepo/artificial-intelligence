import streamlit as st
import numpy as np
import joblib
from PIL import Image
from skimage import color, transform

# Load the trained model
model = joblib.load('digit_classifier_model.pkl')

# Streamlit app
st.title("Digit Classifier")

st.write("Upload a grayscale image of a digit (0-9) to get a prediction.")

uploaded_image = st.file_uploader("Choose an image...", type="png")

if uploaded_image is not None:
    # Open and process the image
    image = Image.open(uploaded_image).convert('L')
    image = np.array(image)

    # Resize the image to 8x8 pixels
    image = transform.resize(image, (8, 8), mode='reflect', anti_aliasing=True)

    # Flatten the image
    image = image.flatten()

    # Predict the digit
    prediction = model.predict([image])

    st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
    st.write(f'Prediction: {prediction[0]}')

# streamlit run app.py
