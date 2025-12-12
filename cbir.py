import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import PIL.Image
import cv2
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from deepface import DeepFace
import torch
from transformers import AutoImageProcessor, AutoModel
import pickle

# Rutas
DB_PATH = Path("cbir_database")
DB_FILE = "image_data.csv"
DB_PATH.mkdir(parents=True, exist_ok=True)
images = Path("images")

# FUNCIONES AUXILIARES
def load_and_preprocess(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # para resnet50
    return x

# EXTRACTORES
class HistogramaColor:
    # vector de 8x8x8 = 512 dimensiones
    def __init__(self, bins=(8, 8, 8)):
        self.bins = bins  # n bins por canal

    def embedding_histograma(self, img_input):
        if isinstance(img_input, (Image.Image, PIL.Image.Image)):
            img_array = np.array(img_input.convert("RGB"))
        else:
            img_path = str(img_input)
            image_cv = cv2.imread(img_path)
            if image_cv is None:
                raise ValueError(f"Error con el path: {img_path}")
            img_array = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        # histograma
        hist = cv2.calcHist([img_array], [0, 1, 2], None, self.bins,
                            [0, 256, 0, 256, 0, 256])
        # normalizar
        hist = cv2.normalize(hist, hist).flatten()

        return hist.astype('float32')  # embedding

    def embeddings_histogramas(self, image_paths):
        embeddings = []
        for path in tqdm(image_paths, desc="Procesando histogramas"):
            try:
                emb = self.embedding_histograma(path)
                embeddings.append(emb)
            except Exception as e:
                print(f"Error con {path}: {e}")
        return np.array(embeddings, dtype='float32') # matriz de embeddings
    
class Resnet50:
    def __init__(self, model_name='resnet50'):
        # 2048-dim embeddings
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.model.trainable = False

    def embedding_resnet(self, img_input):
        if isinstance(img_input, (Image.Image, PIL.Image.Image)):
            img = img_input.resize((224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
        else:
            # Es un path
            img_path = str(img_input)
            x = load_and_preprocess(img_path)
        
        x = preprocess_input(x)
        feat = self.model.predict(x, verbose=0)
        return feat.flatten()  # embedding (1D)

    def embeddings_resnet(self, list_paths, batch_size=32):
        feats = []
        for i in range(0, len(list_paths), batch_size):
            batch_paths = list_paths[i:i+batch_size]
            batch_imgs = []
            for p in batch_paths:
                img = image.load_img(p, target_size=(224,224))
                x = image.img_to_array(img)
                batch_imgs.append(x)
            batch_array = np.array(batch_imgs)
            batch_array = preprocess_input(batch_array)
            batch_feats = self.model.predict(batch_array, verbose=0)
            for f in batch_feats:
                feats.append(f.flatten())
        return np.vstack(feats)
    
class SIFT:
    def __init__(self, n_clusters=100, kmeans_path=None):
        self.n_clusters = n_clusters
        self.sift = cv2.SIFT_create()
        self.kmeans = None
        if kmeans_path and Path(kmeans_path).exists():
            with open(kmeans_path, 'rb') as f:
                self.kmeans = pickle.load(f)

    def extraer_descriptores(self, img_input):
        if isinstance(img_input, (Image.Image, PIL.Image.Image)):
            img_array = np.array(img_input.convert("L"))
        else:
            # Es un path
            img_path = str(img_input)
            image_cv = cv2.imread(img_path)
            if image_cv is None:
                raise ValueError(f"Error en el path: {img_path}")
            img_array = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = self.sift.detectAndCompute(img_array, None)
        if descriptors is None:  # no se detectaron keypoints
            descriptors = np.zeros((1, 128), dtype='float32')
        return descriptors

    def fit_vocabulario(self, image_paths):
        all_descriptors = []
        for path in tqdm(image_paths, desc="Extrayendo SIFT para el vocabulario"):
            desc = self.extraer_descriptores(path)
            all_descriptors.append(desc)
        all_descriptors = np.vstack(all_descriptors)
        # entrenamos kmeans
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(all_descriptors)

    def embedding_sift(self, img_input):
        descriptores = self.extraer_descriptores(img_input)
        
        if self.kmeans is None:
            raise ValueError("KMeans no está entrenado. Llama a fit_vocabulario primero.")
        
        labels = self.kmeans.predict(descriptores)
        # histograma de palabras visuales
        hist, _ = np.histogram(labels, bins=np.arange(self.n_clusters+1))
        hist = hist.astype('float32')
        hist /= (hist.sum() + 1e-7)  # normalizar
        return hist

    def embeddings_sift(self, image_paths):
        embeddings = []
        for path in tqdm(image_paths, desc="Procesando SIFT embeddings"):
            try:
                emb = self.embedding_sift(path)
                embeddings.append(emb)
            except Exception as e:
                print(f"Error con {path}: {e}")
        return np.array(embeddings, dtype='float32')

class VGGFace:
    def __init__(self):
        self.model_name = "VGG-Face"
        DeepFace.build_model(self.model_name)

    def embedding_vgg(self, img_input):
        if isinstance(img_input, (Image.Image, PIL.Image.Image)):
            temp_path = "temp_vgg.jpg"
            img_input.save(temp_path)
            img_path = temp_path
            delete_temp = True
        else:
            img_path = str(img_input)
            delete_temp = False
        
        try:
            embedding_obj = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='opencv',
                align=True
            )
            embedding = embedding_obj[0]["embedding"]
            return np.array(embedding, dtype='float32')
        finally:
            if delete_temp and Path(temp_path).exists():
                Path(temp_path).unlink()

    def embeddings_vgg(self, image_paths):
        embeddings = []
        for path in tqdm(image_paths, desc="Procesando VGG Face"):
            try:
                emb = self.embedding_vgg(path)
                embeddings.append(emb)
            except Exception as e:
                print(f"Error con {path}: {e}")
        return np.array(embeddings, dtype='float32')
    
class Dinov2:
    def __init__(self):
        self.model_name = "facebook/dinov2-base"

        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        # Mover a GPU si esta disponible
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Modelo cargado en: {self.device}")

    def embedding_dino(self, img_input):
        # Si es imagen PIL
        if isinstance(img_input, (Image.Image, PIL.Image.Image)):
            img_pil = img_input.convert("RGB")
        else:
            # Es un path
            img_path = str(img_input)
            img_pil = Image.open(img_path).convert("RGB")

        # Preprocesar para el modelo
        inputs = self.processor(images=img_pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # DINOv2 devuelve el 'last_hidden_state'
        # El primer token contiene la representación global de la imagen
        last_hidden_states = outputs.last_hidden_state
        embedding = last_hidden_states[0, 0, :].cpu().numpy()

        return embedding.astype('float32')

    def embeddings_dino(self, image_paths):
        embeddings = []
        for path in tqdm(image_paths, desc="Procesando DINO"):
            try:
                emb = self.embedding_dino(path)
                embeddings.append(emb)
            except Exception as e:
                print(f"Error con {path}: {e}")
        return np.array(embeddings, dtype='float32')

def build_index(extractor, index_name):
    features = []
    image_files = []
    etiquetas = []
    
    # Recopilar paths de imagenes
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    image_paths = []
    for ext in exts:
        image_paths.extend(images.rglob(ext))
    image_paths = sorted([p for p in image_paths if p.is_file()])

    print(f"Procesando {len(image_paths)} imágenes...")
    
    for img_path in tqdm(image_paths, desc=f"Creando {index_name}"):
        try:
            feat = extractor(img_path)
            features.append(feat)
            image_files.append(img_path.name)

            etiqueta = img_path.parent.name
            etiquetas.append(etiqueta)
        except Exception as e:
            print(f"Error procesando {img_path}: {e}")
    
    if len(features) == 0:
        print("No se procesaron imágenes. Verifica la carpeta 'images'")
        return
    
    features = np.vstack(features).astype(np.float32)
    faiss.normalize_L2(features)
    
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    faiss.write_index(index, str(DB_PATH / index_name))

    # Guardar CSV
    dataframe = pd.DataFrame({'image': image_files, 'etiqueta': etiquetas})
    dataframe.to_csv(DB_PATH / DB_FILE, index=True)
    
    print(f"{index_name} creado con {len(image_files)} imágenes, shape: {features.shape}")

if __name__ == "__main__":
    print("=" * 60)
    print("CREANDO ÍNDICES FAISS")
    print("=" * 60)
    
    # Histograma de color
    print("\n1. Histograma de Color")
    hist_extractor = HistogramaColor()
    build_index(hist_extractor.embedding_histograma, "index_histogram.faiss")

    # ResNet50
    print("\n2. ResNet50")
    resnet_extractor = Resnet50()
    build_index(resnet_extractor.embedding_resnet, "index_resnet.faiss")

    # SIFT
    print("\n3. SIFT")
    sift_extractor = SIFT(n_clusters=100)
    # Primero entrenamos el vocabulario
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    image_paths = []
    for ext in exts:
        image_paths.extend(images.rglob(ext))
    image_paths = sorted([p for p in image_paths if p.is_file()])
    sift_extractor.fit_vocabulario(image_paths)
    
    # Guardar el modelo KMeans
    kmeans_path = DB_PATH / "sift_kmeans.pkl"
    with open(kmeans_path, 'wb') as f:
        pickle.dump(sift_extractor.kmeans, f)
    print(f"KMeans guardado en {kmeans_path}")
    
    build_index(sift_extractor.embedding_sift, "index_sift.faiss")

    # VGG Face
    print("\n4. VGG Face")
    vgg_extractor = VGGFace()
    build_index(vgg_extractor.embedding_vgg, "index_vggface.faiss")

    # DINOv2
    print("\n5. DINOv2")
    dino_extractor = Dinov2()
    build_index(dino_extractor.embedding_dino, "index_dinov2.faiss")
    
    print("\n" + "=" * 60)
    print("TODOS LOS ÍNDICES CREADOS CORRECTAMENTE")
    print("=" * 60)