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

# Función para preprocesar la imagen - versión simplificada y robusta
def preprocess_image(uploaded_file):
    """
    Preprocesa la imagen para que sea compatible con el modelo MNIST
    usando métodos simples pero efectivos.
    """
    # Leer la imagen con PIL (más simple y confiable)
    image = Image.open(uploaded_file).convert('L')  # Convertir a escala de grises
    st.image(image, caption='Imagen Original en Escala de Grises', use_container_width=True)
    
    # Invertir colores si es necesario (MNIST tiene dígitos blancos sobre fondo negro)
    # Calcular el brillo promedio para decidir si invertir
    img_array = np.array(image)
    if np.mean(img_array) < 128:  # Si la imagen es predominantemente oscura
        image = ImageOps.invert(image)
        st.image(image, caption='Imagen con Colores Invertidos', use_container_width=True)
    
    # Aplicar umbral para mejorar el contraste (blanco y negro)
    threshold = 100
    img_array = np.array(image)
    img_binary = np.where(img_array > threshold, 255, 0)
    image = Image.fromarray(img_binary.astype('uint8'))
    st.image(image, caption='Imagen Binarizada', use_container_width=True)
    
    # Recortar bordes blancos para centrar mejor el dígito
    bbox = ImageOps.invert(image).getbbox()
    if bbox:
        image = image.crop(bbox)
        st.image(image, caption='Dígito Recortado', use_container_width=True)
    
    # Crear una nueva imagen cuadrada con fondo negro
    size = max(image.size)
    new_image = Image.new('L', (size, size), 0)
    
    # Pegar la imagen original centrada
    paste_x = (size - image.size[0]) // 2
    paste_y = (size - image.size[1]) // 2
    new_image.paste(image, (paste_x, paste_y))
    st.image(new_image, caption='Dígito Centrado en Imagen Cuadrada', use_container_width=True)
    
    # Redimensionar a 20x20 manteniendo la proporción
    resized_image = new_image.resize((20, 20), Image.LANCZOS)
    
    # Crear la imagen final de 28x28 con el dígito centrado (como en MNIST)
    final_image = Image.new('L', (28, 28), 0)
    paste_x = (28 - 20) // 2
    paste_y = (28 - 20) // 2
    final_image.paste(resized_image, (paste_x, paste_y))
    st.image(final_image, caption='Imagen Final (28x28)', use_container_width=True)
    
    # Convertir a formato numpy y normalizar
    img_array = np.array(final_image)
    img_array = img_array / 255.0
    
    # Reformatear para el modelo (añadir dimensiones de batch y canal)
    model_input = img_array.reshape(1, 28, 28, 1).astype('float32')
    
    return model_input

# Título de la app
st.title('Reconocimiento de Dígitos con CNN - MNIST')

# Espacio para subir la imagen
st.header("Sube una imagen del dígito")
uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG o JPEG)", type=["png", "jpg", "jpeg"])

# Contenedor principal para mostrar imagen y predicción
if uploaded_file is not None:
    # Procesar la imagen
    with st.spinner('Procesando imagen...'):
        processed_image = preprocess_image(uploaded_file)

    if processed_image is not None:
        # Realizar predicción
        with st.spinner('Realizando predicción...'):
            prediction = model.predict(processed_image)
            predicted_label = np.argmax(prediction)
            confidence = np.max(prediction) * 100  # Obtener confianza en porcentaje

        # Mostrar el resultado de la predicción en un formato destacado
        st.subheader("Resultado de la predicción:")
        st.success(f"El dígito predicho es: **{predicted_label}**")
        st.info(f"Confianza del modelo: **{confidence:.2f}%**")

        # Opcional: Mostrar las probabilidades de cada dígito
        if st.checkbox("Mostrar probabilidades por dígito"):
            st.write("Probabilidades para cada dígito:")
            probabilities = prediction[0]
            for i, prob in enumerate(probabilities):
                st.write(f"Dígito {i}: {prob*100:.2f}%")

# Información adicional y del proyecto
st.markdown("---")  # Separador visual

st.header("Información del Proyecto")

with st.expander("Ver detalles del proyecto"):
    st.markdown(""" 
        **Objetivo del Proyecto:**
        El objetivo de este proyecto es desplegar una aplicación basada en Streamlit que permita predecir dígitos escritos a mano utilizando un modelo de red neuronal convolucional (CNN) entrenado con el conjunto de datos MNIST.
        
        Este modelo fue entrenado usando la arquitectura LeNet-5 adaptada para el conjunto de datos MNIST.
        
        **Cómo Usar la Aplicación:**
        1. Sube una imagen que contenga un dígito (debe ser en formato PNG, JPG o JPEG).
        2. La aplicación procesará la imagen y mostrará el dígito predicho por el modelo junto con la confianza.
        3. Opcionalmente, puedes ver las probabilidades que el modelo asignó a cada dígito.
        
        **Características del Preprocesamiento:**
        - Conversión a escala de grises
        - Binarización para mejorar contraste
        - Recorte automático y centrado del dígito
        - Formato compatible con el estándar MNIST (28x28 píxeles)
    """)

st.header("Equipo de Desarrollo")
st.markdown("""
    - Flores Luis
    - Guacanes Kevin
    - Quilca Tatiana
    - Sevilla Masciel
""")

st.markdown("---")  # Otro separador

st.info("Esta aplicación utiliza un modelo de Deep Learning para reconocer dígitos escritos a mano.")