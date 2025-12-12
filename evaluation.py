import os
import time
import pandas as pd
import numpy as np
import faiss
import gc # Garbage Collector para liberar memoria
from tqdm import tqdm
from cbir import HistogramaColor, Resnet50, SIFT, VGGFace, Dinov2

TEST_ROOT = "test"
DB_PATH = "cbir_database"
INDEX_FILES = {
    'Histograma': 'index_histogram.faiss',
    'ResNet50': 'index_resnet.faiss',
    'SIFT': 'index_sift.faiss',
    'VGG-Face': 'index_vggface.faiss',
    'DINOv2': 'index_dinov2.faiss'
}

def get_db_metadata():
    df = pd.read_csv(os.path.join(DB_PATH, 'image_data.csv'))
    return df['etiqueta'].values

# Instancia el modelo solo cuando se pide, para ahorrar RAM
def get_model_instance(name):
    print(f"--> Cargando modelo {name} en memoria...")
    if name == 'Histograma': return HistogramaColor()
    elif name == 'ResNet50': return Resnet50()
    elif name == 'SIFT': return SIFT(kmeans_path=os.path.join(DB_PATH, 'sift_kmeans.pkl'))
    elif name == 'VGG-Face': return VGGFace()
    elif name == 'DINOv2': return Dinov2()
    return None

def evaluate_extractor(name, index_file, test_images):
    print(f"\n=== Evaluando {name} ===")
    
    # 1. Cargar modelo (SOLO AHORA)
    model = get_model_instance(name)
    
    # Seleccionar función de extracción
    if name == 'Histograma': extract_fn = model.embedding_histograma
    elif name == 'ResNet50': extract_fn = model.embedding_resnet
    elif name == 'SIFT': extract_fn = model.embedding_sift
    elif name == 'VGG-Face': extract_fn = model.embedding_vgg
    elif name == 'DINOv2': extract_fn = model.embedding_dino
    
    # 2. Cargar índice FAISS
    index_path = os.path.join(DB_PATH, index_file)
    if not os.path.exists(index_path):
        print(f"Salto: No existe índice {index_file}")
        return []

    index = faiss.read_index(index_path)
    db_labels = get_db_metadata()
    
    results = []

    for img_path, true_label in tqdm(test_images, desc=f"Procesando imágenes"):
        try:
            start_time = time.time()
            
            # Extracción
            embedding = extract_fn(img_path)
            if embedding is None or len(embedding) == 0:
                continue

            # Normalización L2
            vector = np.float32(embedding).reshape(1, -1)
            faiss.normalize_L2(vector)
            
            # Búsqueda FAISS (Top 10)
            _, indices = index.search(vector, k=10)
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            # Métricas
            retrieved_indices = indices[0]
            retrieved_labels = [db_labels[i] for i in retrieved_indices if i < len(db_labels)]
            
            # Top-1 Accuracy
            is_top1_correct = 1 if (len(retrieved_labels) > 0 and retrieved_labels[0] == true_label) else 0
            
            # Top-10 Accuracy (Consistencia)
            matches = sum(1 for label in retrieved_labels if label == true_label)
            top10_accuracy = (matches / 10) * 100
            
            results.append({
                "Famoso": true_label,
                "Extractor": name,
                "Top1_Hit": is_top1_correct,
                "Top10_Acc": top10_accuracy,
                "Time": inference_time
            })

        except Exception as e:
            print(f"Error en imagen: {e}")
            pass
    
    print(f"--> Liberando memoria de {name}...")
    del model
    del index
    gc.collect()
    
    return results

def main():
    test_images = []
    if not os.path.exists(TEST_ROOT):
        print(f"ERROR: No existe la carpeta {TEST_ROOT}")
        return

    for label_dir in os.listdir(TEST_ROOT):
        class_path = os.path.join(TEST_ROOT, label_dir)
        if os.path.isdir(class_path):
            for img in os.listdir(class_path):
                if img.lower().endswith(('.jpg', '.png', '.jpeg')):
                    test_images.append((os.path.join(class_path, img), label_dir))
    
    print(f"Total imágenes de test: {len(test_images)}")
    
    all_data = []
    extractores_list = ['Histograma', 'ResNet50', 'SIFT', 'VGG-Face', 'DINOv2']
    
    for name in extractores_list:
        if name in INDEX_FILES:
            idx_file = INDEX_FILES[name]
            data = evaluate_extractor(name, idx_file, test_images)
            all_data.extend(data)
        
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print("No se generaron resultados.")
        return
    
    # Tabla A: Resumen Global
    print("\n" + "="*50)
    print("TABLA 1: RESUMEN GLOBAL (Para el Paper)")
    print("="*50)
    summary = df.groupby("Extractor").agg({
        "Top1_Hit": lambda x: (sum(x)/len(x))*100, 
        "Top10_Acc": "mean",                        
        "Time": "mean"                              
    }).sort_values("Top10_Acc", ascending=False)
    
    summary = summary.rename(columns={
        "Top1_Hit": "Top-1 Accuracy (%)", 
        "Top10_Acc": "Top-10 Accuracy (%)", 
        "Time": "Tiempo Medio (s)"
    })
    print(summary)
    summary.to_csv("tabla1_global.csv")

    # Tabla B: Desglose
    print("\n" + "="*50)
    print("TABLA 2: DESGLOSE POR FAMOSO (Top-10 Acc)")
    print("="*50)
    pivot = df.pivot_table(
        index="Famoso", 
        columns="Extractor", 
        values="Top10_Acc", 
        aggfunc="mean"
    )
    print(pivot)
    pivot.to_csv("tabla2_famosos.csv")
    
    print("\n¡Éxito! Datos guardados en CSV.")

if __name__ == "__main__":
    main()