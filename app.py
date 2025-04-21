import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import imageio

# Cargar el modelo
model = load_model('models/model_Mnist_LeNet.h5')

# Función para preprocesar la imagen
def preprocess_image(image_path):
    img = imageio.imread(image_path)
    gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])  # Convertir a escala de grises
    gray = gray.reshape(1, 28, 28, 1)
    gray = gray.astype(np.float32) / 255.0
    return gray

# Título de la app
st.title('Reconocimiento de Dígitos con CNN - MNIST')

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Mostrar la imagen
    img = imageio.imread(uploaded_file)
    st.image(img, caption='Imagen Subida', use_column_width=True)

    # Preprocesar imagen
    processed_image = preprocess_image(uploaded_file)

    # Realizar predicción
    prediction = model.predict(processed_image)
    predicted_label = np.argmax(prediction)

    st.write(f"El dígito predicho es: {predicted_label}")

# Instrucciones
st.sidebar.markdown("""
    **Instrucciones:**
    1. Sube una imagen del dígito que deseas reconocer.
    2. La aplicación mostrará la imagen y el dígito predicho.
""")
