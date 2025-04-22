import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import io

# Configurar la página
st.set_page_config(page_title="Reconocimiento de Dígitos MNIST", layout="centered")

# Cargar el modelo (asegúrate de que la ruta sea correcta)
try:
    model = load_model('models/model_Mnist_LeNet.h5')
    st.success("Modelo cargado exitosamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()  # Detiene la ejecución si no se carga el modelo

# Función para preprocesar la imagen - versión simplificada
def preprocess_image(uploaded_file):
    """
    Lee una imagen subida, la convierte a escala de grises, la redimensiona
    y la prepara para la entrada al modelo.
    """
    st.write("Paso 1: Cargando la imagen...")
    image = Image.open(uploaded_file).convert('L')  # Convertir a escala de grises
    st.image(image, caption='Imagen Original', use_container_width=True)

    # Invertir colores si es necesario (MNIST tiene dígitos blancos sobre fondo negro)
    img_array = np.array(image)
    if np.mean(img_array) > 128:  # Si la imagen es predominantemente clara
        st.write("Paso 2: Invirtiendo colores para ajustar al formato MNIST...")
        image = ImageOps.invert(image)
        st.image(image, caption='Imagen con Colores Invertidos', use_container_width=True)

    # Redimensionar si no es 28x28
    if image.size != (28, 28):
        st.write("Paso 3: Redimensionando la imagen a 28x28 píxeles...")
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        st.image(image, caption='Imagen Redimensionada a 28x28', use_container_width=True)

    # Normalizar la imagen para el modelo
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalizar a valores entre 0 y 1
    
    # Reformatear para la entrada del modelo
    img_array = img_array.reshape(1, 28, 28, 1)
    st.write("Imagen normalizada para la entrada del modelo.")
    
    return img_array

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
            confidence = np.max(prediction) * 100  # Obtener confianza en porcentaje
            st.write("Predicción realizada.")

        # Mostrar el resultado de la predicción en un formato destacado
        st.subheader("Resultado de la predicción:")
        st.success(f"El dígito predicho es: **{predicted_label}**")
        st.info(f"Confianza del modelo: **{confidence:.2f}%**")

        # Mostrar las probabilidades de cada dígito
        st.subheader("Probabilidades para cada dígito:")
        probabilities = prediction[0]
        
        # Crear un gráfico de barras
        chart_data = {str(i): float(prob) for i, prob in enumerate(probabilities)}
        st.bar_chart(chart_data)
        
        # Mostrar también los valores numéricos
        st.write("Valores numéricos de probabilidad:")
        for i, prob in enumerate(probabilities):
            st.write(f"Dígito {i}: {prob*100:.2f}%")

# Información adicional y del proyecto (ahora en el cuerpo principal)
st.markdown("---") # Separador visual

st.header("Información del Proyecto")

with st.expander("Ver detalles del proyecto"):
    st.markdown(""" 
        **Objetivo del Proyecto:**
        El objetivo de este proyecto es desplegar una aplicación basada en Streamlit que permita predecir dígitos escritos a mano utilizando un modelo de red neuronal convolucional (CNN) entrenado con el conjunto de datos MNIST.
        
        Este modelo fue entrenado usando la arquitectura LeNet-5 adaptada para el conjunto de datos MNIST.
        
        **Cómo Usar la Aplicación:**
        1. Sube una imagen que contenga un dígito (debe ser en formato PNG, JPG o JPEG).
        2. La aplicación procesará la imagen y mostrará el dígito predicho por el modelo junto con la confianza.
        3. Puedes ver las probabilidades que el modelo asignó a cada dígito.
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