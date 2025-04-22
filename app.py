import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
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

# Función para preprocesar la imagen - versión extremadamente robusta
def preprocess_image(uploaded_file):
    """
    Preprocesa cualquier tipo de imagen para hacerla compatible con el modelo MNIST,
    incluso imágenes de muy baja calidad o con estilos muy diferentes.
    """
    # Leer la imagen con PIL
    image = Image.open(uploaded_file)
    
    # Mostrar imagen original
    st.image(image, caption='Imagen Original', use_container_width=True)
    
    # Convertir a escala de grises si no lo está ya
    if image.mode != 'L':
        image = image.convert('L')
        st.image(image, caption='Imagen en Escala de Grises', use_container_width=True)
    
    # Aplicar filtros de mejora
    # 1. Aumentar contraste
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)  # Aumentar contraste al doble
    
    # 2. Suavizar para reducir ruido
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # 3. Aplicar umbral adaptativo
    # Primero convertimos a array numpy para manipulación más flexible
    img_array = np.array(image)
    
    # Calcular umbral dinámico basado en el histograma (método Otsu simplificado)
    hist = np.histogram(img_array, bins=256, range=(0, 256))[0]
    total = hist.sum()
    sumB = 0
    wB = 0
    maximum = 0
    threshold = 0
    
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (total - sumB) / wF
        
        between = wB * wF * (mB - mF) ** 2
        
        if between > maximum:
            maximum = between
            threshold = i
    
    # Aplicar umbral para binarizar
    img_binary = np.where(img_array > threshold, 255, 0).astype(np.uint8)
    image = Image.fromarray(img_binary)
    st.image(image, caption='Imagen Binarizada con Umbral Adaptativo', use_container_width=True)
    
    # Invertir colores si es necesario (MNIST tiene dígitos blancos sobre fondo negro)
    # Verificamos la distribución de píxeles para decidir
    pixel_sum = np.sum(img_binary)
    total_pixels = img_binary.size
    
    # Si hay más pixeles blancos que negros, invertimos
    if pixel_sum > (total_pixels * 255 * 0.5):
        image = ImageOps.invert(image)
        st.image(image, caption='Imagen con Colores Invertidos', use_container_width=True)
    
    # Eliminar pequeños ruidos (operación morfológica de apertura)
    image = image.filter(ImageFilter.MinFilter(3))
    image = image.filter(ImageFilter.MaxFilter(3))
    st.image(image, caption='Imagen con Ruido Reducido', use_container_width=True)
    
    # Recortar bordes blancos para centrar mejor el dígito
    # Usamos invert para encontrar el bounding box
    bbox = ImageOps.invert(image).getbbox()
    
    if bbox:
        image = image.crop(bbox)
        st.image(image, caption='Dígito Recortado', use_container_width=True)
    else:
        st.warning("No se pudo detectar un contorno claro. Usando imagen completa.")
    
    # Agregar un pequeño padding
    border = 10
    width, height = image.size
    new_img = Image.new('L', (width + 2*border, height + 2*border), 0)
    new_img.paste(image, (border, border))
    image = new_img
    
    # Crear una nueva imagen cuadrada
    size = max(image.size)
    square_img = Image.new('L', (size, size), 0)
    
    # Pegar la imagen original centrada
    paste_x = (size - image.size[0]) // 2
    paste_y = (size - image.size[1]) // 2
    square_img.paste(image, (paste_x, paste_y))
    st.image(square_img, caption='Dígito Centrado en Imagen Cuadrada', use_container_width=True)
    
    # Redimensionar a 20x20 manteniendo la proporción
    resized_img = square_img.resize((20, 20), Image.Resampling.LANCZOS)
    
    # Crear la imagen final de 28x28 con el dígito centrado (como en MNIST)
    final_img = Image.new('L', (28, 28), 0)
    paste_x = (28 - 20) // 2
    paste_y = (28 - 20) // 2
    final_img.paste(resized_img, (paste_x, paste_y))
    st.image(final_img, caption='Imagen Final (28x28)', use_container_width=True)
    
    # Mejorar imagen final con filtro de nitidez
    final_img = final_img.filter(ImageFilter.SHARPEN)
    st.image(final_img, caption='Imagen Final con Nitidez Mejorada', use_container_width=True)
    
    # Convertir a formato numpy y normalizar
    img_array = np.array(final_img)
    img_array = img_array / 255.0
    
    # Reformatear para el modelo (añadir dimensiones de batch y canal)
    model_input = img_array.reshape(1, 28, 28, 1).astype('float32')
    
    return model_input

# Función para mostrar las probabilidades como gráfico de barras
def display_prediction_chart(prediction):
    # Crear datos para el gráfico
    probs = prediction[0] * 100  # Convertir a porcentaje
    
    # Crear diccionario para el gráfico
    chart_data = {str(i): float(prob) for i, prob in enumerate(probs)}
    
    # Mostrar gráfico
    st.bar_chart(chart_data)

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

        # Mostrar probabilidades como gráfico de barras
        st.subheader("Probabilidades por dígito:")
        display_prediction_chart(prediction)
        
        # Opción para ver los valores exactos
        if st.checkbox("Mostrar valores numéricos de probabilidad"):
            st.write("Probabilidades detalladas:")
            for i, prob in enumerate(prediction[0]):
                st.write(f"Dígito {i}: {prob*100:.2f}%")

# Información adicional y del proyecto
st.markdown("---")  # Separador visual

st.header("Información del Proyecto")

with st.expander("Ver detalles del proyecto"):
    st.markdown(""" 
        **Cómo Usar la Aplicación:**
        1. Sube una imagen que contenga un dígito (debe ser en formato PNG, JPG o JPEG).
        2. La aplicación procesará la imagen y mostrará el dígito predicho por el modelo junto con la confianza.
        3. Puedes ver las probabilidades que el modelo asignó a cada dígito mediante un gráfico de barras.
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