import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
from scipy import ndimage
import matplotlib.pyplot as plt

# Configurar la página
st.set_page_config(page_title="Reconocimiento de Dígitos MNIST", layout="centered")

# Cargar el modelo (asegúrate de que la ruta sea correcta)
try:
    model = load_model('models/model_Mnist_LeNet.h5')
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop() # Detiene la ejecución si no se carga el modelo

def preprocess_image(uploaded_file):
    """
    Lee una imagen subida, la convierte a escala de grises, la redimensiona
    y la prepara para la entrada al modelo.
    """
    try:
        # Leer la imagen desde el archivo subido
        img_bytes = uploaded_file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convertir a escala de grises
        img = img.convert('L')
        
        # Mostrar la imagen original convertida a escala de grises
        st.image(img, caption='Imagen Original en Escala de Grises', width=200)
        
        # Redimensionar a 28x28 pixels
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convertir a array de numpy
        gray = np.array(img)
        
        # Mostrar la imagen redimensionada
        st.image(gray, caption='Imagen Redimensionada a 28x28', width=150)
        
        # Normalizar a [0,1]
        gray = gray.astype(np.float32) / 255.0
        
        # Invertir si es necesario (MNIST tiene fondo negro y dígitos blancos)
        if gray.mean() > 0.5:  # Si la imagen es mayormente blanca
            gray = 1.0 - gray
            st.image(gray, caption='Imagen Invertida', width=150)
        
        # Centrar el dígito usando centro de masa
        cy, cx = ndimage.center_of_mass(gray)
        rows, cols = gray.shape
        shift_y = rows//2 - cy
        shift_x = cols//2 - cx
        
        # Aplicar desplazamiento para centrar
        gray = ndimage.shift(gray, (shift_y, shift_x), cval=0)
        st.image(gray, caption='Imagen Centrada (Formato MNIST)', width=150)
        
        # Redimensionar a formato adecuado para el modelo
        gray = gray.reshape(1, 28, 28, 1)
        
        return gray
        
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
        return None

# Título de la app
st.title('Reconocimiento de Dígitos con CNN - MNIST')

# Espacio para subir la imagen
st.header("Sube una imagen del dígito")
uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG o JPEG)", type=["png", "jpg", "jpeg"])

# Contenedor principal para mostrar imagen y predicción
if uploaded_file is not None:
    # Crear una copia para mostrar la imagen original
    uploaded_file_copy = uploaded_file.copy()
    
    # Mostrar la imagen original
    st.image(uploaded_file_copy, caption='Imagen Original Subida', use_container_width=True)
    
    # Procesar la imagen
    with st.spinner('Procesando imagen...'):
        processed_image = preprocess_image(uploaded_file)

    if processed_image is not None:
        # Realizar predicción
        with st.spinner('Realizando predicción...'):
            prediction = model.predict(processed_image)
            predicted_label = np.argmax(prediction)
            confidence = np.max(prediction) * 100 # Obtener confianza en porcentaje

        # Mostrar el resultado de la predicción en un formato destacado
        st.subheader("Resultado de la predicción:")
        st.success(f"El dígito predicho es: {predicted_label}")
        st.info(f"Confianza del modelo: {confidence:.2f}%")

        # Opcional: Mostrar las probabilidades de cada dígito
        if st.checkbox("Mostrar probabilidades por dígito"):
            st.write("Probabilidades para cada dígito:")
            probabilities = prediction[0]
            # Crear un gráfico de barras para visualizar las probabilidades
            fig, ax = plt.subplots(figsize=(10, 5))
            digits = np.arange(10)
            ax.bar(digits, probabilities)
            ax.set_xticks(digits)
            ax.set_xlabel('Dígito')
            ax.set_ylabel('Probabilidad')
            ax.set_title('Probabilidad por dígito')
            st.pyplot(fig)
            
            # También mostrar como valores numéricos
            prob_dict = {str(i): f"{prob*100:.2f}%" for i, prob in enumerate(probabilities)}
            st.json(prob_dict)


# Información adicional y del proyecto
st.markdown("---") # Separador visual

st.header("Información del Proyecto")

with st.expander("Ver detalles del proyecto"):
    st.markdown("""
        Objetivo del Proyecto:
        El objetivo de este proyecto es desplegar una aplicación basada en Streamlit que permita predecir dígitos escritos a mano utilizando un modelo de red neuronal convolucional (CNN) entrenado con el conjunto de datos MNIST.

        Este modelo fue entrenado usando la arquitectura LeNet-5 adaptada para el conjunto de datos MNIST.

        Cómo Usar la Aplicación:
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