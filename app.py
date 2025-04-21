import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import io
import cv2

# Configurar la página
st.set_page_config(page_title="Reconocimiento de Dígitos MNIST", layout="centered")

# Cargar el modelo (asegúrate de que la ruta sea correcta)
try:
    model = load_model('models/model_Mnist_LeNet.h5')
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop() # Detiene la ejecución si no se carga el modelo

# Función para preprocesar la imagen
def preprocess_image(uploaded_file):
    """
    Lee una imagen subida, la convierte a escala de grises, encuentra el bounding box del dígito,
    redimensiona a 28x28 y la prepara para la entrada al modelo.
    """
    # Leer la imagen con OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Mostrar la imagen original en escala de grises
    st.subheader("Imagen en escala de grises")
    st.image(gray, caption='Imagen en escala de grises', use_column_width=True)
    
    # Aplicar umbral para binarizar la imagen (números negros sobre fondo blanco)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Encontrar contornos para detectar el dígito
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Encontrar el contorno más grande (asumiendo que es el dígito)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Obtener el bounding box
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Dibujar el bounding box en la imagen original
        bbox_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(bbox_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Mostrar la imagen con el bounding box
        st.subheader("Bounding Box detectado")
        st.image(bbox_img, caption='Bounding Box del dígito', use_column_width=True)
        
        # Extraer solo el área del dígito con un pequeño margen
        margin = 5
        x_min = max(0, x - margin)
        y_min = max(0, y - margin)
        x_max = min(gray.shape[1], x + w + margin)
        y_max = min(gray.shape[0], y + h + margin)
        
        digit_roi = binary[y_min:y_max, x_min:x_max]
        
        # Mostrar la región de interés (ROI) del dígito
        st.subheader("Región de interés (ROI)")
        st.image(digit_roi, caption='ROI del dígito', use_column_width=True)
        
        # Redimensionar a 28x28 para el modelo MNIST
        digit_resized = cv2.resize(digit_roi, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Mostrar la imagen redimensionada
        st.subheader("Imagen procesada (28x28)")
        st.image(digit_resized, caption='Imagen redimensionada a 28x28', use_column_width=True)
        
        # Preparar para el modelo
        processed = digit_resized.reshape(1, 28, 28, 1)
        processed = processed.astype(np.float32) / 255.0
        
        return processed
    else:
        st.error("No se detectaron contornos en la imagen. Asegúrate de que el dígito sea visible.")
        return None

# Título de la app
st.title('Reconocimiento de Dígitos con CNN - MNIST')

# Espacio para subir la imagen
st.header("Sube una imagen del dígito")
uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG o JPEG)", type=["png", "jpg", "jpeg"])

# Contenedor principal para mostrar imagen y predicción
if uploaded_file is not None:
    # Mostrar la imagen original
    image_data = uploaded_file.getvalue()
    uploaded_file.seek(0)  # Reiniciar el puntero del archivo para usarlo nuevamente
    st.subheader("Imagen Original")
    st.image(image_data, caption='Imagen Subida', use_column_width=True)

    # Procesar la imagen
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
             # Crear un diccionario o lista de tuplas para facilitar la visualización
             prob_dict = {str(i): float(prob) for i, prob in enumerate(probabilities)}
             st.json(prob_dict)


# Información adicional y del proyecto (ahora en el cuerpo principal)
st.markdown("---") # Separador visual

st.header("Información del Proyecto")

with st.expander("Ver detalles del proyecto"):
    st.markdown("""
        Objetivo del Proyecto:
        El objetivo de este proyecto es desplegar una aplicación basada en Streamlit que permita predecir dígitos escritos a mano utilizando un modelo de red neuronal convolucional (CNN) entrenado con el conjunto de datos MNIST.

        Este modelo fue entrenado usando la arquitectura LeNet-5 adaptada para el conjunto de datos MNIST.

        Cómo Usar la Aplicación:
        1. Sube una imagen que contenga un dígito (debe ser en formato PNG, JPG o JPEG).
        2. La aplicación procesará la imagen, la convertirá a escala de grises, detectará el bounding box del dígito y lo redimensionará a 28x28.
        3. El modelo analizará la imagen procesada y mostrará el dígito predicho junto con la confianza.
        4. Opcionalmente, puedes ver las probabilidades que el modelo asignó a cada dígito.
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