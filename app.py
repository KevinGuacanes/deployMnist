import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import imageio.v2 as imageio
from PIL import Image
from skimage.transform import resize
from scipy import ndimage
import matplotlib.pyplot as plt

# Configurar la página
st.set_page_config(page_title="Reconocimiento de Dígitos MNIST", layout="centered")

# Cargar el modelo (asegúrate de que la ruta sea correcta)
try:
    model = load_model('models/model_Mnist_LeNet.h5')
    st.success("Modelo cargado exitosamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop() # Detiene la ejecución si no se carga el modelo

# Función para preprocesar la imagen
def preprocess_image(uploaded_file):
    """
    Preprocesa una imagen para que coincida con el formato MNIST.
    
    Incluye:
    - Conversión a escala de grises
    - Redimensionamiento a 28x28
    - Normalización de valores
    - Inversión automática si es necesario
    - Centrado mediante el centro de masa
    """
    st.write("Paso 1: Cargando la imagen...")
    im = imageio.imread(uploaded_file)
    st.image(im, caption='Imagen Original', use_container_width=True)
    
    # Convertir a escala de grises si es necesario
    if len(im.shape) > 2:
        st.write("Paso 2: Convirtiendo la imagen a escala de grises...")
        gray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
    else:
        st.write("La imagen ya está en escala de grises.")
        gray = im
    
    st.image(gray, caption='Imagen en Escala de Grises', use_container_width=True)
    
    # Redimensionar a 28x28
    st.write("Paso 3: Redimensionando la imagen a 28x28 píxeles...")
    gray = resize(gray, (28, 28), anti_aliasing=True)
    st.image(gray, caption='Imagen Redimensionada a 28x28', use_container_width=True)
    
    # Normalizar los valores a [0,1]
    gray = gray.astype('float32')
    gray /= 255.0
    
    # Invertir la imagen si es necesario (si el fondo es más oscuro que el dígito)
    if gray.mean() > 0.5:
        st.write("Paso 4: Invirtiendo la imagen (fondo claro, dígito oscuro)...")
        gray = 1.0 - gray
        st.image(gray, caption='Imagen Invertida', use_container_width=True)
    
    # Agregar padding y centrar la imagen mediante el centro de masa
    st.write("Paso 5: Centrando el dígito usando el centro de masa...")
    cy, cx = ndimage.center_of_mass(gray)
    rows, cols = gray.shape
    shift_y = rows // 2 - cy
    shift_x = cols // 2 - cx
    gray = ndimage.shift(gray, (shift_y, shift_x), cval=0)
    st.image(gray, caption='Imagen Centrada', use_container_width=True)
    
    # Redimensionar para el modelo
    st.write("Paso 6: Preparando la imagen para el modelo...")
    gray_for_model = gray.reshape(1, 28, 28, 1)
    
    return gray_for_model

# Título de la app
st.title('Reconocimiento de Dígitos con CNN - MNIST')

# Espacio para subir la imagen
st.header("Sube una imagen del dígito")
uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG o JPEG)", type=["png", "jpg", "jpeg"])

# Contenedor principal para mostrar imagen y predicción
if uploaded_file is not None:
    # Procesar la imagen
    processed_image = preprocess_image(uploaded_file)

    if processed_image is not None:
        # Realizar predicción
        with st.spinner('Realizando predicción...'):
            prediction = model.predict(processed_image)
            predicted_label = np.argmax(prediction)
            confidence = np.max(prediction) * 100 # Obtener confianza en porcentaje
            st.write("Predicción realizada.")

        # Mostrar el resultado de la predicción en un formato destacado
        st.subheader("Resultado de la predicción:")
        st.success(f"El dígito predicho es: **{predicted_label}**")
        st.info(f"Confianza del modelo: **{confidence:.2f}%**")

        # Opcional: Mostrar las probabilidades de cada dígito
        if st.checkbox("Mostrar probabilidades por dígito"):
             st.write("Probabilidades para cada dígito:")
             probabilities = prediction[0]
             # Crear un diccionario o lista de tuplas para facilitar la visualización
             prob_dict = {str(i): f"{prob*100:.2f}%" for i, prob in enumerate(probabilities)}
             st.json(prob_dict)

# Información adicional y del proyecto (ahora en el cuerpo principal)
st.markdown("---") # Separador visual

st.header("Información del Proyecto")

with st.expander("Ver detalles del proyecto"):
    st.markdown(""" 
        **Objetivo del Proyecto:**
        El objetivo de este proyecto es desplegar una aplicación basada en Streamlit que permita predecir dígitos escritos a mano utilizando un modelo de red neuronal convolucional (CNN) entrenado con el conjunto de datos MNIST.
        
        Este modelo fue entrenado usando la arquitectura LeNet-5 adaptada para el conjunto de datos MNIST.
        
        **Características del Preprocesamiento:**
        - Conversión a escala de grises
        - Redimensionamiento a 28x28 píxeles
        - Normalización de valores
        - Inversión automática (si el fondo es más claro que el dígito)
        - Centrado automático del dígito mediante el centro de masa
        
        **Cómo Usar la Aplicación:**
        1. Sube una imagen que contenga un dígito (debe ser en formato PNG, JPG o JPEG).
        2. La aplicación procesará la imagen y mostrará el dígito predicho por el modelo junto con la confianza.
        3. Opcionalmente, puedes ver las probabilidades que el modelo asignó a cada dígito.
    """)

st.header("Equipo de Desarrollo")
st.markdown("""
    - Flores Luis
    - Guacanes Kevin
    - Quilca Tatiana
    - Sevilla Masciel
""")

st.markdown("---") # Otro separador

st.info("Esta aplicación utiliza un modelo de Deep Learning para reconocer dígitos escritos a mano.")