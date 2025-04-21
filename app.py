import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import imageio
from PIL import Image
import cv2

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
    Lee una imagen subida, la convierte a escala de grises, la redimensiona
    y la prepara para la entrada al modelo con métodos mejorados para preservar la calidad.
    """
    # Leer la imagen como array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Mostrar imagen original
    st.write("Paso 1: Imagen original cargada")
    # Convertir de BGR a RGB para visualización correcta en Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption='Imagen Original', use_container_width=True)
    
    # Convertir a escala de grises usando CV2 (mejor control)
    st.write("Paso 2: Convirtiendo a escala de grises con preservación de detalles")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    st.image(gray, caption='Imagen en Escala de Grises', use_container_width=True)
    
    # Aplicar mejora de contraste a través de ecualización de histograma
    st.write("Paso 3: Mejorando contraste para destacar el dígito")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    st.image(enhanced, caption='Imagen con Contraste Mejorado', use_container_width=True)
    
    # Aplicar filtro para reducir ruido pero preservar bordes (opcional según calidad de entrada)
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Redimensionar usando un mejor algoritmo
    st.write("Paso 4: Redimensionando a 28x28 con preservación de características")
    # Encontrar contorno del dígito para centrar
    ret, thresh = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Si encontramos contornos, trabajamos con el área principal
    if contours:
        # Encontrar el contorno más grande (debería ser el dígito)
        c = max(contours, key=cv2.contourArea)
        
        # Obtener el cuadro delimitador
        x, y, w, h = cv2.boundingRect(c)
        
        # Extraer el dígito y añadir un pequeño margen
        margin = 5
        y_min = max(0, y - margin)
        y_max = min(enhanced.shape[0], y + h + margin)
        x_min = max(0, x - margin)
        x_max = min(enhanced.shape[1], x + w + margin)
        
        digit = enhanced[y_min:y_max, x_min:x_max]
        
        # Mostrar el dígito extraído
        st.image(digit, caption='Dígito Extraído', use_container_width=True)
        
        # Crear imagen cuadrada (necesario para mantener proporción)
        side_length = max(digit.shape[0], digit.shape[1])
        square_img = np.zeros((side_length, side_length), dtype=np.uint8)
        
        # Centrar el dígito en la imagen cuadrada
        y_offset = (side_length - digit.shape[0]) // 2
        x_offset = (side_length - digit.shape[1]) // 2
        square_img[y_offset:y_offset+digit.shape[0], x_offset:x_offset+digit.shape[1]] = digit
        
        # Redimensionar a 20x20 para mantener espacio en blanco alrededor (como en MNIST)
        resized = cv2.resize(square_img, (20, 20), interpolation=cv2.INTER_AREA)
        
        # Crear imagen final 28x28 con el dígito centrado
        final_img = np.zeros((28, 28), dtype=np.uint8)
        offset = (28 - 20) // 2
        final_img[offset:offset+20, offset:offset+20] = resized
        
    else:
        # Si no se detectaron contornos, simplemente redimensionar la imagen completa
        st.warning("No se detectó claramente un dígito, procesando la imagen completa.")
        final_img = cv2.resize(enhanced, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Mostrar imagen final procesada
    st.image(final_img, caption='Imagen Final Procesada (28x28)', use_container_width=True)
    
    # Invertir colores si es necesario para que coincida con MNIST (dígitos blancos sobre fondo negro)
    if np.mean(final_img) > 127:
        final_img = 255 - final_img
        st.image(final_img, caption='Imagen con Colores Invertidos', use_container_width=True)
    
    # Normalizar para el modelo (0-1)
    normalized = final_img.astype('float32') / 255.0
    
    # Reformatear para la entrada del modelo
    model_input = normalized.reshape(1, 28, 28, 1)
    
    st.success("Preprocesamiento completado con éxito")
    return model_input

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

        # Mostrar el resultado de la predicción en un formato destacado
        st.subheader("Resultado de la predicción:")
        st.success(f"El dígito predicho es: **{predicted_label}**")
        st.info(f"Confianza del modelo: **{confidence:.2f}%**")

        # Opcional: Mostrar las probabilidades de cada dígito
        if st.checkbox("Mostrar probabilidades por dígito"):
            st.write("Probabilidades para cada dígito:")
            probabilities = prediction[0]
            # Crear un gráfico de barras para mejor visualización
            chart_data = {str(i): float(prob) for i, prob in enumerate(probabilities)}
            st.bar_chart(chart_data)

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
        3. Opcionalmente, puedes ver las probabilidades que el modelo asignó a cada dígito.
        
        **Características del Preprocesamiento:**
        - Extracción automática del dígito de la imagen
        - Centrado del dígito en la imagen
        - Mejora de contraste adaptativo
        - Redimensionamiento preservando características importantes
        - Formato compatible con el estándar MNIST (28x28 píxeles)
    """)

st.header("Equipo de Desarrollo")
st.markdown("""
    - Flores Luis
    - Guacanes Kevin
    - Quilca Tatiana
    - Sevilla Masciel
""")

st.markdown("---") # Otro separador

st.info("Esta aplicación utiliza un modelo de Deep Learning para reconocer dígitos escritos a mano. La calidad del preprocesamiento mejora significativamente la precisión del reconocimiento.")