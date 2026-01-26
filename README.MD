# CBIR IMAGE SEARCH – Content-Based Image Retrieval

> Reidentificación de celebridades mediante visión artificial.

Este proyecto implementa un sistema de Recuperación de Imágenes Basado en Contenido (CBIR) diseñado para la reidentificación de personas. Utiliza un pipeline modular en Python que integra **FAISS** para la indexación vectorial de alta velocidad y permite comparar el rendimiento de 5 extractores de características que representan la evolución del estado del arte en visión artificial.

El sistema incluye una interfaz interactiva basada en **Streamlit** y un módulo de evaluación automatizada.

## 🚀 Guía de Inicio Rápido
### 1. Preparación de los Datos
1.  **Descarga el *dataset* de galería (images):** [Enlace a Google Drive](https://drive.google.com/drive/folders/1gIFuspPFP5dRvDbXX0FnwGBM5fhbfAz3?usp=sharing)
2.  **Descarga el *dataset* de consulta (test):** [Enlace a Google Drive](https://drive.google.com/drive/folders/1jYbeSWFMlA3aM-rnECpdqRfAL51p2NjO?usp=sharing)
3.  **Estructura de carpetas:** Descomprime los archivos y asegúrate de que tu proyecto tenga esta estructura exacta:

    ```text
    /proyecto
    ├── /images              # Imágenes de Galería (Base de conocimiento)
    │   ├── /Angelina Jolie
    │   ├── /Leonardo DiCaprio
    │   └── ...
    │
    └── /test                # Imágenes de Consulta (Para pruebas)
        ├── /Angelina Jolie
        └── ...
    ```

### 2. Instalación de Dependencias
1. **Crear el entorno**

En la terminal Windows, dentro de la carpeta del proyecto:

```bash
python -m venv venv
venv\Scripts\activate
```

En la terminal Linux o Mac, dentro de la carpeta del proyecto:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. **Instalar requisitos**

```bash
pip install -r requirements.txt
```

### 3. Generación de Índices FAISS

**Nota: la carpeta ```/cbir_database``` que viene dada en este repositorio corresponde con los índices FAISS de nuestro conjunto de datos de galería. Si es el conjunto con el que quieres hacer la búsqueda, puedes saltarte este paso.**

Antes de realizar búsquedas, es necesario procesar las imágenes de la galería y generar los índices vectoriales .faiss.

```bash
python cbir.py
```

Este comando:
- Recorre la carpeta ```/images```
- Extrae *embeddings* usando los 5 extractores (Histograma, SIFT, ResNet50, VGG-Face, DINOv2).
- Normaliza los vectores
- Genera los índices en la carpeta ```/cbir_database```
- Genera el archivo de metadatos ```/cbir_database/image_data.csv```

### 4. Interfaz de Usuario Streamlit

Para interactuar visualmente con el motor de búsqueda y realizar consultas manuales:

```bash
streamlit run app.py
```
La aplicación se abrirá automáticamente en ```http://localhost:8501```.

Funcionalidades:
- Selección dinámica de extractor.
- Carga y recorte manual de la imagen de consulta.
- Visualización de los resultados más similares, imágenes y etiquetas.
- Cálculo de Top-10 *Accuracy* en tiempo real (si se introduce la etiqueta).

### 5. Evaluación automatizada (benchmarking)

**Nota: se han incluido los archivos resultantes de esta parte en la carpeta ```/result``` realizados con nuestro conjunto de consulta. Si desea evaluar el mismo, puede saltarse este paso.**

Para obtener las métricas de rendimiento (Top-k *Accuracy* y Latencia) sobre todo el conjunto de test sin intervención manual:

```bash
python evaluation.py
```

Este comando:
- Recorre la carpeta ```/test```
- Computa el Top-k *Accuracy* medio para $k=1$ y $k=10$ y latencia media para cada extractor, así como el Top-10 Accuracy medio para cada par famoso-extractor.
- Genera dos archivos ```.csv``` en la carpeta ```/proyecto``` con las métricas detalladas

## 🧠 Extractores Implementados
| Modelo | Tipo | Descripción | Dimensión Vector |
| :--- | :--- | :--- | :--- |
| **Histograma de Color** | Estadístico | Distribución de color RGB. | 512 |
| **SIFT** | Descriptor Local | BoVW + K-Means. | 100 |
| **ResNet50** | CNN (Generalista) | Pre-entrenada con ImageNet. | 2048 |
| **VGG-Face** | CNN (Especialista) | *Fine-tuned* para reconocimiento facial. | 4096 |
| **DINOv2** | *Transformer* | Modelo autosupervisado (ViT) de Meta AI. | 768 |

## 📂 Estructura del proyecto
```text
cbir_project/
│
├── app.py
├── cbir.py
├── evaluation.py
├── requirements.txt
│
├── cbir_database/
│   ├── index_resnet.faiss
│   ├── index_vggface.faiss
│   ├── image_data.csv
│   └── ...
│
├── images/ 
├── test/
└── result/
```

## 📝 Notas importantes
- El código se ha probado en dispositivos con RAM >= 16GB. Hay modelos que pueden aprovechar CUDA si está disponible, pero los resultados generados en el *benchmark* no serán válidos al no aplicarse a todos los modelos.
- No se recomienda editar manualmente los archivos ```.faiss```.
- Si se añaden nuevas imágenes a la carpeta ```/images``` se deberá de ejecutar de nuevo ```python cbir.py``` para considerar los nuevos índices en la galería.

---

**Autores**: Laura Martín y Miguel Gallego


**Fecha**: 12 Dec 2025

<div align="center">
    <img src="https://www.upm.es/gsfs/SFS11864" width="500" alt="Logo UPM">
</div>
