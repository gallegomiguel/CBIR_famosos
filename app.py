import time
import torch
import faiss
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import os

import streamlit as st
from streamlit_cropper import st_cropper

# Importar los extractores desde cbir.py
from cbir import HistogramaColor, Resnet50, SIFT, VGGFace, Dinov2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(layout="wide")

device = torch.device('cpu')

FILES_PATH = str(pathlib.Path().resolve())

# Path in which the images should be located
IMAGES_PATH = os.path.join(FILES_PATH, 'images')
# Path in which the database should be located
DB_PATH = os.path.join(FILES_PATH, 'cbir_database')

DB_FILE = 'image_data.csv'

# Diccionario para cachear los modelos cargados
_model_cache = {}

def get_model(extractor_name):
    if extractor_name not in _model_cache:
        if extractor_name == 'Histograma de Color':
            _model_cache[extractor_name] = HistogramaColor()
        elif extractor_name == 'ResNet50':
            _model_cache[extractor_name] = Resnet50()
        elif extractor_name == 'SIFT':
            kmeans_path = os.path.join(DB_PATH, 'sift_kmeans.pkl')
            _model_cache[extractor_name] = SIFT(n_clusters=100, kmeans_path=kmeans_path)
        elif extractor_name == 'VGG-Face':
            _model_cache[extractor_name] = VGGFace()
        elif extractor_name == 'DINOv2':
            _model_cache[extractor_name] = Dinov2()
    return _model_cache[extractor_name]

def get_image_list():
    df = pd.read_csv(os.path.join(DB_PATH, DB_FILE))
    image_list = list(df.image.values)
    etiquetas = list(df.etiqueta.values)
    return image_list, etiquetas

def retrieve_image(img_query, feature_extractor, n_imgs=11):
    # Seleccionar el modelo y el índice según el extractor
    if feature_extractor == 'Histograma de Color':
        model = get_model('Histograma de Color')
        embeddings = model.embedding_histograma(img_query)
        indexer = faiss.read_index(os.path.join(DB_PATH, 'index_histogram.faiss'))
        
    elif feature_extractor == 'ResNet50':
        model = get_model('ResNet50')
        embeddings = model.embedding_resnet(img_query)
        indexer = faiss.read_index(os.path.join(DB_PATH, 'index_resnet.faiss'))
        
    elif feature_extractor == 'SIFT':
        model = get_model('SIFT')
        embeddings = model.embedding_sift(img_query)
        indexer = faiss.read_index(os.path.join(DB_PATH, 'index_sift.faiss'))
        
    elif feature_extractor == 'VGG-Face':
        model = get_model('VGG-Face')
        embeddings = model.embedding_vgg(img_query)
        indexer = faiss.read_index(os.path.join(DB_PATH, 'index_vggface.faiss'))
        
    elif feature_extractor == 'DINOv2':
        model = get_model('DINOv2')
        embeddings = model.embedding_dino(img_query)
        indexer = faiss.read_index(os.path.join(DB_PATH, 'index_dinov2.faiss'))

    # Normalizar y buscar
    vector = np.float32(embeddings).reshape(1, -1)
    faiss.normalize_L2(vector)

    _, indices = indexer.search(vector, k=n_imgs)

    return indices[0]

def calculate_accuracy(retrieved_indices, query_label, etiquetas, k=10):
    # Tomar solo las primeras k imágenes (excluyendo la primera que es la query misma)
    top_k_indices = retrieved_indices[1:k+1]
    
    # Contar cuántas tienen la misma etiqueta
    matches = sum(1 for idx in top_k_indices if etiquetas[idx] == query_label)
    
    accuracy = (matches / k) * 100
    return accuracy

def main():
    st.title('CBIR IMAGE SEARCH')
    
    col1, col2 = st.columns(2)

    with col1:
        st.header('QUERY')

        st.subheader('Choose feature extractor')
        option = st.selectbox('.', ('Histograma de Color', 'ResNet50', 'SIFT', 'VGG-Face', 'DINOv2'))

        st.subheader('Upload image')
        img_file = st.file_uploader(label='.', type=['png', 'jpg'])

        if img_file:
            img = Image.open(img_file)
            # Get a cropped image from the frontend
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')
            
            # Manipulate cropped image at will
            st.write("Preview")
            _ = cropped_img.thumbnail((150,150))
            st.image(cropped_img)

    with col2:
        st.header('RESULT')
        if img_file:
            st.markdown('**Retrieving .......**')
            start = time.time()

            retriev = retrieve_image(cropped_img, option, n_imgs=11)
            image_list, etiquetas = get_image_list()

            end = time.time()
            st.markdown('**Finish in ' + str(end - start) + ' seconds**')

            col3, col4 = st.columns(2)

            with col3:
                img_path = os.path.join(IMAGES_PATH, etiquetas[retriev[0]], image_list[retriev[0]])
                image = Image.open(img_path)
                st.image(image, use_container_width=True)
                st.caption(f"Etiqueta: {etiquetas[retriev[0]]}")

            with col4:
                img_path = os.path.join(IMAGES_PATH, etiquetas[retriev[1]], image_list[retriev[1]])
                image = Image.open(img_path)
                st.image(image, use_container_width=True)
                st.caption(f"Etiqueta: {etiquetas[retriev[1]]}")

            col5, col6, col7 = st.columns(3)

            with col5:
                for u in range(2, 11, 3):
                    img_path = os.path.join(IMAGES_PATH, etiquetas[retriev[u]], image_list[retriev[u]])
                    image = Image.open(img_path)
                    st.image(image, use_container_width=True)
                    st.caption(f"Etiqueta: {etiquetas[retriev[u]]}")

            with col6:
                for u in range(3, 11, 3):
                    img_path = os.path.join(IMAGES_PATH, etiquetas[retriev[u]], image_list[retriev[u]])
                    image = Image.open(img_path)
                    st.image(image, use_container_width=True)
                    st.caption(f"Etiqueta: {etiquetas[retriev[u]]}")

            with col7:
                for u in range(4, 11, 3):
                    img_path = os.path.join(IMAGES_PATH, etiquetas[retriev[u]], image_list[retriev[u]])
                    image = Image.open(img_path)
                    st.image(image, use_container_width=True)
                    st.caption(f"Etiqueta: {etiquetas[retriev[u]]}")

            # Calculo de accuracy
            st.markdown("---")
            st.subheader("Accuracy Calculation")
            query_label_input = st.text_input("Enter the label of the query image (optional):", "")
            
            if query_label_input:
                accuracy = calculate_accuracy(retriev, query_label_input, etiquetas, k=10)
                st.metric("Top-10 Accuracy", f"{accuracy:.1f}%")
                st.caption("Percentage of the top 10 retrieved images that match the query label")

if __name__ == '__main__':
    main()