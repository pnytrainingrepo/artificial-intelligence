import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import transform

# Load the trained CNN model
model = tf.keras.models.load_model('cnn_digit_classifier_model.h5')

# Streamlit app
st.title("Digit Classifier with CNN")

st.write("Upload a grayscale image of a digit (0-9) to get a prediction.")

uploaded_image = st.file_uploader("Choose an image...", type="png")

if uploaded_image is not None:
    # Open and process the image
    image = Image.open(uploaded_image).convert('L')
    image = np.array(image)

    # Resize the image to 28x28 pixels
    image = transform.resize(image, (28, 28), mode='reflect', anti_aliasing=True)

    # Expand dimensions to match the model input
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)  # (1, 28, 28, 1)

    # Predict the digit
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=-1)[0]

    st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
    st.write(f'Prediction: {predicted_class}')

# streamlit run app.py