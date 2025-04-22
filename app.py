import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import io

# Configurar la página con mejor apariencia
st.set_page_config(
    page_title="Reconocimiento de Dígitos MNIST", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Título principal con estilo
st.markdown("""
    <h1 style='text-align: center; color: #2e6c80;'>Reconocimiento de Dígitos con CNN - MNIST</h1>
    <p style='text-align: center; color: #5e5e5e;'>Una aplicación de Machine Learning para reconocer dígitos escritos a mano</p>
    <hr>
""", unsafe_allow_html=True)

# Cargar el modelo (asegúrate de que la ruta sea correcta)
@st.cache_resource
def load_mnist_model():
    try:
        return load_model('models/model_Mnist_LeNet.h5')
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_mnist_model()
if model is None:
    st.stop()  # Detener la ejecución si no se carga el modelo
else:
    st.success("✅ Modelo cargado exitosamente.")

# Función para preprocesar la imagen - versión secuencial (hacia abajo)
def preprocess_image(uploaded_file):
    """
    Lee una imagen subida, la convierte a escala de grises, la redimensiona
    y la prepara para la entrada al modelo.
    """
    # Mostrar la imagen original
    st.subheader("1️⃣ Imagen Original")
    image = Image.open(uploaded_file).convert('L')  # Convertir a escala de grises
    st.image(image, caption='Imagen subida convertida a escala de grises', width=300)

    # Invertir colores si es necesario (MNIST tiene dígitos blancos sobre fondo negro)
    img_array = np.array(image)
    
    st.subheader("2️⃣ Imagen Procesada")
    if np.mean(img_array) > 128:  # Si la imagen es predominantemente clara
        image = ImageOps.invert(image)
        st.info("Se invirtieron los colores para ajustar al formato MNIST")
    
    # Redimensionar si no es 28x28
    if image.size != (28, 28):
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        st.info("Imagen redimensionada a 28x28 píxeles")
    
    st.image(image, caption='Imagen lista para el modelo', width=300)

    # Normalizar la imagen para el modelo
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalizar a valores entre 0 y 1
    
    # Reformatear para la entrada del modelo
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

# Sección de carga de imagen
st.subheader("📤 Cargar Imagen")
uploaded_file = st.file_uploader("Selecciona una imagen de un dígito", type=["png", "jpg", "jpeg"])

# Contenedor principal para mostrar imagen y predicción
if uploaded_file is not None:
    # Procesar la imagen
    processed_image = preprocess_image(uploaded_file)

    if processed_image is not None:
        # Realizar predicción
        with st.spinner('Analizando imagen...'):
            prediction = model.predict(processed_image)
            predicted_label = np.argmax(prediction)
            confidence = np.max(prediction) * 100  # Obtener confianza en porcentaje

        # Mostrar resultados en una sección destacada
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: #2e6c80;'>Resultados del Análisis</h2>", unsafe_allow_html=True)
        
        # Mostrar el dígito predicho en una caja destacada
        st.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; width: 50%; margin: 0 auto;'>
            <h1 style='font-size: 80px; color: #2e6c80;'>{predicted_label}</h1>
            <p style='font-size: 18px;'>Dígito Predicho</p>
            <p style='font-size: 16px; color: #5e5e5e;'>Confianza: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar las probabilidades
        st.subheader("Distribución de Probabilidades")
        
        # Crear un gráfico de barras para las probabilidades
        chart_data = {str(i): float(prob) for i, prob in enumerate(prediction[0])}
        st.bar_chart(chart_data)
        
        # Mostrar la tabla de probabilidades
        st.subheader("Tabla de Probabilidades")
        prob_df = {"Dígito": list(range(10)), "Probabilidad (%)": [p*100 for p in prediction[0]]}
        st.dataframe(prob_df, use_container_width=True)

# Información del proyecto en tarjetas organizadas
st.markdown("---")

# Crear pestañas para organizar la información adicional
tab1, tab2, tab3 = st.tabs(["📋 Información del Proyecto", "👨‍💻 Equipo", "❓ Ayuda"])

with tab1:
    st.markdown("""
    ### Objetivo del Proyecto
    
    Este proyecto implementa una aplicación basada en Streamlit que permite reconocer dígitos escritos a mano utilizando una Red Neuronal Convolucional (CNN) entrenada con el conjunto de datos MNIST.
    
    ### Características
    
    - **Arquitectura:** LeNet-5 adaptada para el conjunto de datos MNIST
    - **Preprocesamiento de imágenes:** Conversión a escala de grises, redimensionamiento e inversión de colores cuando sea necesario
    - **Visualización:** Muestra las probabilidades para cada dígito y la confianza del modelo
    """)

with tab2:
    st.markdown("""
    ### Integrantes
    
    - **Flores Luis**
    - **Guacanes Kevin**
    - **Quilca Tatiana**
    - **Sevilla Masciel**
    """)

with tab3:
    st.markdown("""
    ### Cómo Usar la Aplicación
    
    1. **Carga una imagen** que contenga un solo dígito escrito a mano (formatos: PNG, JPG o JPEG)
    2. La aplicación **procesará automáticamente** la imagen para que sea compatible con el formato MNIST
    3. El modelo realizará la **predicción** y mostrará el dígito reconocido
    4. Podrás ver la **distribución de probabilidades** que el modelo asignó a cada dígito
    
    ### Mejores Prácticas
    
    - Usa imágenes con fondos claros y dígitos oscuros para mejores resultados
    - Si los resultados no son precisos, intenta mejorar la calidad de la imagen o usar un dígito más claramente definido
    - El modelo funciona mejor con dígitos escritos a mano similares al estilo del dataset MNIST
    """)