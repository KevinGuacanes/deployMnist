import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
# No es necesario importar image de tensorflow.keras.preprocessing para este caso
import matplotlib.pyplot as plt # Aunque no se usa en el código proporcionado, lo mantengo por si acaso.
import imageio # Usaremos imageio para leer la imagen

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
    Lee una imagen subida, la convierte a escala de grises, la redimensiona
    y la prepara para la entrada al modelo.
    """
    # Usar imageio para leer la imagen del archivo BytesIO
    img = imageio.imread(uploaded_file)

    # Asegurarse de que la imagen es RGB si tiene 3 canales antes de convertir a gris
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = np.dot(img[...,:3], [0.299, 0.587, 0.114]) # Convertir a escala de grises
    elif len(img.shape) == 3 and img.shape[2] == 4: # Si es RGBA
         gray = np.dot(img[...,:3], [0.299, 0.587, 0.114]) # Ignorar canal alfa
    elif len(img.shape) == 2: # Si ya está en escala de grises
        gray = img
    else:
         st.error("Formato de imagen no soportado.")
         return None

    # Redimensionar la imagen si no es 28x28
    if gray.shape != (28, 28):
        # Nota: Una redimensión simple puede distorsionar la imagen.
        # Para mejor precisión, considera usar OpenCV o Pillow con interpolación.
        # Aquí usamos un enfoque básico de numpy para mantener las dependencias mínimas.
        # Si necesitas redimensionar, puedes usar:
        # from PIL import Image
        # pil_img = Image.fromarray(gray.astype(np.uint8))
        # pil_img_resized = pil_img.resize((28, 28), Image.Resampling.LANCZOS) # o LUMINANCE, BILINEAR, etc.
        # gray = np.array(pil_img_resized)
        st.warning("La imagen no es 28x28 píxeles. No se realizará la redimensión automática.")
        # Si decides redimensionar aquí, descomenta y usa la parte de PIL.
        # Por ahora, si no es 28x28, podrías tener errores o resultados incorrectos
        # a menos que tu modelo maneje diferentes tamaños. Asumimos 28x28 es requerido.
        # Para este ejemplo, detendremos el proceso si no es 28x28.
        return None


    gray = gray.reshape(1, 28, 28, 1) # Añadir dimensiones de batch y canal
    gray = gray.astype(np.float32) / 255.0 # Normalizar
    return gray

# Título de la app
st.title('Reconocimiento de Dígitos con CNN - MNIST')

# Espacio para subir la imagen
st.header("Sube una imagen del dígito")
uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG o JPEG)", type=["png", "jpg", "jpeg"])

# Contenedor principal para mostrar imagen y predicción
if uploaded_file is not None:
    # Mostrar la imagen
    st.image(uploaded_file, caption='Imagen Subida', use_container_width=True)

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
             # Crear un diccionario o lista de tuplas para facilitar la visualización
             prob_dict = {str(i): prob for i, prob in enumerate(probabilities)}
             st.json(prob_dict)


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
    """)

st.header("Equipo de Desarrollo")
st.markdown("""
    - Flores Luis
    - Guacanes Kevin
    - Quilca Tatiana
    - Sevilla Masciel
""")

st.markdown("---") # Otro separador

st.info("Esta aplicación utiliza un modelo de Deep Learning para reconocer dígitos escritos a mano. ")