# CBIR IMAGE SEARCH – Content-Based Image Retrieval

Este proyecto implementa un sistema CBIR utilizando FAISS y múltiples extractores de características, accesible desde una interfaz web creada con Streamlit.

El usuario puede subir una imagen que se recortará automáticamente y obtener las imágenes más similares dentro de la base de datos según el extractor seleccionado.

Incluye además un cálculo opcional de Accuracy para evaluar el rendimiento del sistema.

## PASOS A SEGUIR PARA EJECUTARLO:
### 1. Descarga del dataset de imágenes

Las imágenes del proyecto están disponibles en Google Drive:

Enlace de la base de datos:
https://drive.google.com/drive/folders/1gIFuspPFP5dRvDbXX0FnwGBM5fhbfAz3?usp=sharing

Cómo preparar las imágenes
    - Descarga el ZIP desde el enlace.
    - Descomprímelo.
    - Coloca la carpeta resultante dentro del proyecto, asegurándote de que la estructura quede así:
        /images
            /Angelina Jolie
                img1.jpg
                img2.jpg
            /Leonardo DiCaprio
                img3.jpg
                img4.jpg
            ...

Cada subcarpeta es la etiqueta utilizada para los cálculos y para mostrar resultados.

Adjuntamos también nuestra carpeta con la que hemos lanzado las queries a la aplicación:
https://drive.google.com/drive/folders/1jYbeSWFMlA3aM-rnECpdqRfAL51p2NjO?usp=sharing

Esto se trata de otra carpeta que utilizaremos para comprobar el funcionamiento del sistema.

### 2. Instalación de dependencias

Crea un entorno virtual:

python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

Instala los requisitos:
pip install -r requirements.txt

### 3. Generar los índices FAISS

Antes de usar la app, se deben crear los índices y para ello hay que ejecutar:
python cbir.py

Esto:
- Extrae embeddings usando los 5 extractores
- Normaliza los vectores
- Construye índices FAISS
- Genera el archivo cbir_database/image_data.csv
- Guarda el modelo KMeans para SIFT

### 4. Ejecutar la aplicación Streamlit

Una vez generados los índices, ejecutamos en la terminal:

streamlit run app.py

La aplicación se abrirá en:

http://localhost:8501

## Extractores disponibles
1. Histograma de Color. Comparación basada en colores (rápido)
2. ResNet50. CNN pre-entrenada, 2048 dimensiones
3. SIFT + Bag of Visual Words. Características locales + KMeans
4. VGG-Face. Optimizado para detección de rostros
5. DINOv2 (Meta AI). State-of-the-art en visión self-supervised

## Funcionamiento de la aplicación

En la columna izquierda:
- Selección del extractor
- Cargar imagen
- Recortar manualmente la imagen y ver previsualización

En la columna derecha:
- Recuperación de las 10 imágenes más similares
- Visualización organizada en columnas
- Tiempo de inferencia
- Cálculo de Top-10 Accuracy si introduces la etiqueta de la query (Escribir el nombre del famoso)

## Cálculo de Accuracy

Puedes introducir la etiqueta correcta de la imagen de consulta: Top-10 Accuracy = (nº coincidencias entre las 10 mejores) / 10

## Notas importantes
- La carpeta cbir_database/ se genera automáticamente.
- No editar a mano los archivos .faiss.
- Si se añaden imágenes nuevas, vuelve a ejecutar cbir.py para reconstruir los índices.
- Para DINOv2, con GPU CUDA, el proceso será más rápido.

## Estructura del proyecto
cbir/
│
├── app.py                # Interfaz Streamlit
├── cbir.py               # Extractores y construcción de índices
│
├── cbir_database/        # Se genera automáticamente
│   ├── index_resnet.faiss
│   ├── index_sift.faiss
│   ├── index_histogram.faiss
│   ├── index_vggface.faiss
│   ├── index_dinov2.faiss
│   └── image_data.csv
│
└── images/               # Carpeta con las imágenes (descargadas desde Drive)
    └── <carpetas_por_etiqueta>/<imagen>.jpg
