import os
import zipfile
import kaggle

# Descargar el dataset usando Kaggle API
def download_dataset():
    # Usar la API de Kaggle para descargar y descomprimir automáticamente el dataset
    kaggle.api.dataset_download_files('paultimothymooney/chest-xray-pneumonia', path='datasets', unzip=True)

# Descomprimir el dataset (ya no es necesario porque Kaggle API puede descomprimir automáticamente)
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Organizar el dataset en carpetas
def organize_dataset(data_path, output_path):
    categories = ['train', 'test']
    subcategories = ['NORMAL', 'PNEUMONIA']
    
    # Crear la estructura de carpetas si no existen
    for category in categories:
        for subcategory in subcategories:
            os.makedirs(os.path.join(output_path, category, subcategory), exist_ok=True)
    
    # Mover archivos a la estructura de carpetas adecuada
    for category in categories:
        for subcategory in subcategories:
            folder_path = os.path.join(data_path, category, subcategory)
            for filename in os.listdir(folder_path):
                src = os.path.join(folder_path, filename)
                dst = os.path.join(output_path, category, subcategory, filename)
                if os.path.isfile(src):
                    os.rename(src, dst)

# Descargar y organizar el dataset
download_dataset()
data_path = 'datasets/chest_xray'
output_path = 'organized_chest_xray'
organize_dataset(data_path, output_path)

print("Descarga y organización del dataset completadas.")
