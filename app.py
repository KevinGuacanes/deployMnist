import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import io
from skimage.transform import resize
from scipy import ndimage

# Configurar la página con un diseño más compacto
st.set_page_config(
    page_title="Reconocimiento de Dígitos MNIST", 
    layout="centered",  # Cambiado a "centered" para un diseño más compacto
    initial_sidebar_state="collapsed"
)

# Título principal con estilo pero más compacto
st.markdown("""
    <h1 style='text-align: center; color: #2e6c80;'>Reconocimiento de Dígitos MNIST</h1>
    <p style='text-align: center; color: #5e5e5e;'>Aplicación ML para reconocer dígitos escritos a mano</p>
    <hr>
""", unsafe_allow_html=True)

# Cargar el modelo con caché para mejorar rendimiento
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

# Función para preprocesar la imagen con técnicas mejoradas
def preprocess_image(uploaded_file):
    """
    Preprocesa una imagen para que coincida con el formato MNIST.
    Incluye conversión a escala de grises, redimensionamiento,
    inversión automática y centrado mediante centro de masa.
    """
    # Leer la imagen
    image = Image.open(uploaded_file).convert('L')  # Convertir a escala de grises
    img_array = np.array(image)
    
    # Mostrar la imagen original
    st.subheader("Imagen Original")
    st.image(image, caption='Imagen original en escala de grises', width=200)
    
    # Invertir colores si es necesario (MNIST tiene dígitos blancos sobre fondo negro)
    if np.mean(img_array) > 128:  # Si la imagen es predominantemente clara
        image = ImageOps.invert(image)
        img_array = np.array(image)
        st.info("Se invirtieron los colores para ajustar al formato MNIST")
        st.image(image, caption='Imagen con colores invertidos', width=200)
    
    # Redimensionar a 28x28
    if image.size != (28, 28):
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(image)
        st.info("Imagen redimensionada a 28x28 píxeles")
    
    # Normalizar la imagen
    img_array = img_array.astype('float32') / 255.0
    
    # Centrar el dígito usando el centro de masa
    cy, cx = ndimage.center_of_mass(img_array)
    rows, cols = img_array.shape
    shift_y = rows // 2 - cy
    shift_x = cols // 2 - cx
    
    if abs(shift_x) > 2 or abs(shift_y) > 2:  # Solo centrar si el desplazamiento es significativo
        img_array = ndimage.shift(img_array, (shift_y, shift_x), cval=0)
        # Convertir de nuevo a imagen para mostrar
        centered_image = Image.fromarray((img_array * 255).astype(np.uint8))
        st.info("Dígito centrado usando centro de masa")
        st.image(centered_image, caption='Imagen centrada y lista para el modelo', width=200)
    
    # Reformatear para la entrada del modelo
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

# Sección para cargar la imagen
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
        
        # Resultado del dígito predicho
        st.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; max-width: 300px; margin: 0 auto;'>
            <h1 style='font-size: 60px; color: #2e6c80;'>{predicted_label}</h1>
            <p style='font-size: 16px;'>Dígito Predicho</p>
            <p style='font-size: 14px; color: #5e5e5e;'>Confianza: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Distribución de probabilidades en un formato más compacto
        st.subheader("Distribución de Probabilidades")
        
        # Crear un gráfico de barras para las probabilidades
        chart_data = {str(i): float(prob) for i, prob in enumerate(prediction[0])}
        st.bar_chart(chart_data)
        
        # Tabla de probabilidades en formato compacto
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dígitos 0-4**")
            for i in range(5):
                st.write(f"Dígito {i}: {prediction[0][i]*100:.2f}%")
        
        with col2:
            st.markdown("**Dígitos 5-9**")
            for i in range(5, 10):
                st.write(f"Dígito {i}: {prediction[0][i]*100:.2f}%")

# Información del proyecto en formato de pestañas más compactas
st.markdown("---")

# Crear pestañas para organizar la información adicional
tab1, tab2, tab3 = st.tabs(["📋 Información", "👨‍💻 Equipo", "❓ Ayuda"])

with tab1:
    st.markdown("""
    Este proyecto implementa una aplicación que permite reconocer dígitos escritos a mano utilizando una Red Neuronal Convolucional (CNN) entrenada con el conjunto de datos MNIST.
    
    **Características:**
    - Arquitectura LeNet-5 adaptada
    - Preprocesamiento avanzado de imágenes
    - Centrado automático mediante centro de masa
    """)

with tab2:
    st.markdown("""
    **Integrantes:**
    - Flores Luis
    - Guacanes Kevin
    - Quilca Tatiana
    - Sevilla Masciel
    """)

with tab3:
    st.markdown("""
    **Uso de la Aplicación:**
    1. Carga una imagen de un dígito escrito a mano
    2. La aplicación procesará y centrará la imagen
    3. Visualiza el resultado de la predicción y las probabilidades
    
    **Consejo:** Usa imágenes con dígitos claros y fondos simples para mejores resultados.
    """)

# Pie de página simple
st.markdown("""
<div style='text-align: center; color: #888; padding: 10px; font-size: 12px;'>
    Proyecto de Reconocimiento de Dígitos MNIST utilizando Deep Learning
</div>
""", unsafe_allow_html=True)