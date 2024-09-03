import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import transform

# Load the trained CNN model
model = tf.keras.models.load_model('lung_cancer_cnn_model.h5')

# Streamlit app
st.title("Lung Cancer Detection")

st.write("Upload a lung scan image to predict if it shows signs of cancer.")

uploaded_image = st.file_uploader("Choose an image...", type="png")

if uploaded_image is not None:
    # Open and process the image
    image = Image.open(uploaded_image).convert('RGB')
    image = np.array(image)

    # Resize the image to 150x150 pixels
    image = transform.resize(image, (150, 150), mode='reflect', anti_aliasing=True)

    # Expand dimensions to match the model input
    image = np.expand_dims(image, axis=0)  # (1, 150, 150, 3)

    # Predict the class
    prediction = model.predict(image)
    predicted_class = 'Cancer' if prediction[0] > 0.5 else 'No Cancer'

    st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
    st.write(f'Prediction: {predicted_class}')

# streamlit run app.py