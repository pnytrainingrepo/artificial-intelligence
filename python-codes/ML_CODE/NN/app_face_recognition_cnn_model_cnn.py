import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import transform
from sklearn.preprocessing import LabelEncoder

# Load the trained CNN model
model = tf.keras.models.load_model('face_recognition_cnn_model.h5')

# Define the label encoder for the target names
target_names = ['Person 1', 'Person 2', 'Person 3']  # Update with actual target names

# Streamlit app
st.title("Face Recognition")

st.write("Upload an image of a face to get recognition.")

uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:
    # Open and preprocess the image
    image = Image.open(uploaded_image).convert('RGB')
    image = np.array(image)

    # Resize the image to match the model input
    image = transform.resize(image, (input_shape[0], input_shape[1]), mode='reflect', anti_aliasing=True)

    # Add channel dimension and expand dimensions to match model input
    image = np.expand_dims(image, axis=-1) if len(image.shape) == 3 else np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)  # (1, height, width, channels)

    # Predict the class
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=-1)[0]

    st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
    st.write(f'Prediction: {target_names[predicted_class]}')

# streamlit run app.py