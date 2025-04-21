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
def preprocess_image(uploaded_file):
    img = imageio.imread(uploaded_file)
    gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])  # Convertir a escala de grises
    gray = gray.reshape(1, 28, 28, 1)
    gray = gray.astype(np.float32) / 255.0
    return gray

# Título de la app
st.title('Reconocimiento de Dígitos con CNN - MNIST')

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen del dígito (en formato PNG, JPG o JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Mostrar la imagen
    img = imageio.imread(uploaded_file)
    st.image(img, caption='Imagen Subida', use_container_width=True)

    # Preprocesar imagen
    processed_image = preprocess_image(uploaded_file)

    # Realizar predicción
    prediction = model.predict(processed_image)
    predicted_label = np.argmax(prediction)

    # Mostrar el resultado de la predicción
    st.write(f"El dígito predicho es: {predicted_label}")

# Sección de información adicional
st.sidebar.header("Información sobre el proyecto")
st.sidebar.markdown("""
    **Equipo de Desarrollo:**
    - Flores Luis
    - Guacanes Kevin
    - Quilca Tatiana
    - Sevilla Masciel

    **Objetivo del Proyecto:**
    El objetivo de este proyecto es desplegar una aplicación basada en Streamlit que permita predecir dígitos escritos a mano utilizando un modelo de red neuronal convolucional (CNN) entrenado con el conjunto de datos MNIST.

    **Cómo Usar la Aplicación:**
    1. Subir una imagen que contenga un dígito (debe ser en formato PNG, JPG o JPEG).
    2. La aplicación procesará la imagen y mostrará el dígito predicho por el modelo.
""")

# Instrucciones
st.sidebar.markdown("""
    **Instrucciones:**
    1. Sube una imagen de un dígito para realizar la predicción.
    2. La aplicación mostrará la imagen junto con el dígito predicho.
""")

# Footer con agradecimientos
st.markdown("---")
st.markdown("""
    **Agradecimientos:**
    - Este proyecto fue desarrollado por estudiantes de la Universidad [Nombre de la Universidad].
    - Gracias por utilizar esta aplicación de predicción de dígitos con redes neuronales convolucionales.
""")
