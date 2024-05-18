Claro, aquí tienes el README.md con los pasos estructurados pero sin incluir el código de `main.py` ni `download_and_prepare_data.py`.

```markdown
# Clasificación de Radiografías de Neumonía

Este proyecto clasifica radiografías de tórax para detectar neumonía utilizando una red neuronal convolucional (CNN). Incluye la descarga del dataset desde Kaggle, el preprocesamiento de los datos, el entrenamiento del modelo y la implementación de una interfaz gráfica para la predicción.

## Prerrequisitos

1. **Instalar Python**: Asegúrate de tener Python instalado en tu sistema. Puedes descargarlo desde [python.org](https://www.python.org/).

2. **Instalar Git**: Asegúrate de tener Git instalado en tu sistema. Puedes descargarlo desde [git-scm.com](https://www.git-scm.com/).

3. **Crear y Activar un Entorno Virtual (opcional, pero recomendado)**:
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # En Windows usa `myenv\Scripts\activate`
    ```

## Paso 1: Clonar el Repositorio

Clona el repositorio del proyecto desde GitHub (asegúrate de tener el repositorio subido a GitHub):

```bash
git clone <URL_del_repositorio>
cd <nombre_del_repositorio>
```

## Paso 2: Crear el Archivo .gitignore

Crea un archivo .gitignore en el directorio raíz del proyecto con el siguiente contenido:

```
# Ignorar la carpeta de datasets
datasets/

# Ignorar la carpeta organized_chest_xray
organized_chest_xray/
```

## Paso 3: Instalar las Dependencias

Instala las dependencias necesarias desde el archivo requirements.txt. Crea este archivo en el directorio raíz del proyecto y añade las siguientes líneas:

```
numpy
opencv-python
Pillow
scikit-learn
tensorflow
kaggle
```

Luego, instala las dependencias usando pip:

```bash
pip install -r requirements.txt
```

## Paso 4: Configurar Kaggle

1. **Crear una Cuenta en Kaggle**: Si no tienes una cuenta en Kaggle, crea una en [kaggle.com](https://www.kaggle.com/).

2. **Generar la API Key**:
    - Ve a la sección de tu perfil en Kaggle y selecciona "Account".
    - Desplázate hacia abajo y selecciona "Create New API Token". Esto descargará un archivo `kaggle.json`.

3. **Configurar la API Key**:
    - Mueve el archivo `kaggle.json` al directorio `~/.kaggle/` (crea este directorio si no existe).

    ```bash
    mkdir -p ~/.kaggle
    mv ~/Descargas/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    ```

## Paso 5: Descargar y Organizar el Dataset

Crea un script `download_and_prepare_data.py` con el siguiente contenido para descargar y organizar el dataset.

Ejecuta el script para descargar y organizar el dataset:

```bash
python download_and_prepare_data.py
```

## Paso 6: Ejecutar el Aplicativo

Crea un script `main.py` para ejecutar el aplicativo.

Para ejecutar el programa, simplemente ejecuta:

```bash
python main.py
```
```