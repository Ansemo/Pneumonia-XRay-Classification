import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import Label
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Función para cargar datos
def load_data(data_dir, img_size=(128, 128)):
    images = []
    labels = []
    # Iterar sobre las carpetas de cada clase de enfermedad
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            # Iterar sobre cada archivo de imagen en la carpeta de la clase
            for file in os.listdir(label_path):
                img_path = os.path.join(label_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append(label)
    # Normalizar las imágenes
    images = np.array(images) / 255.0
    labels = np.array(labels)
    return images, labels

# Cargar datos de entrenamiento
train_data_dir = 'organized_chest_xray/train'  # Cambia esta ruta a la ubicación de tus datos
img_size = (128, 128)
images, labels = load_data(train_data_dir, img_size)

# Imprimir la cantidad de imágenes cargadas
print(f"Imágenes cargadas: {len(images)}")
print(f"Etiquetas cargadas: {len(labels)}")

# Verificar si hay imágenes y etiquetas cargadas
if len(labels) == 0:
    raise ValueError("No se encontraron imágenes en el directorio especificado.")

# Convertir etiquetas a formato categórico
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Añadir una dimensión para el canal de color (1 canal para escala de grises)
images = np.expand_dims(images, axis=-1)

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

# Crear el modelo de CNN
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(np.unique(labels)), activation='softmax')
    ])
    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Definir la forma de entrada para el modelo
input_shape = (img_size[0], img_size[1], 1)
model_path = 'chest_xray_model.h5'

# Cargar el modelo si existe, de lo contrario, entrenar uno nuevo
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = create_model(input_shape)
    # Entrenar el modelo con los datos de entrenamiento
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    # Guardar el modelo entrenado
    model.save(model_path)

# Función para predecir la enfermedad a partir de una imagen
def predict_disease(img_path, model, img_size=(128, 128)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # Añadir dimensión para el canal de color
    img = np.expand_dims(img, axis=0)  # Añadir dimensión para el lote
    # Realizar la predicción con el modelo
    prediction = model.predict(img)
    # Obtener la etiqueta predicha
    predicted_label = le.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# Crear la interfaz gráfica
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificación de Radiografías")
        
        self.frame = ttk.Frame(root, padding="20")
        self.frame.pack(fill="both", expand=True)
        
        self.label = ttk.Label(self.frame, text="Seleccione una imagen de radiografía")
        self.label.pack(pady=10)
        
        self.btn_select = ttk.Button(self.frame, text="Seleccionar Imagen", command=self.load_image)
        self.btn_select.pack(pady=10)
        
        self.panel = ttk.Label(self.frame)  # Panel para mostrar la imagen
        self.panel.pack(pady=10)
        
        self.result = ttk.Label(self.frame, text="")
        self.result.pack(pady=10)
    
    def load_image(self):
        img_path = filedialog.askopenfilename()
        if img_path:
            try:
                # Cargar y mostrar la imagen seleccionada
                img = Image.open(img_path)
                img = img.resize((250, 250), Image.LANCZOS)
                img = ImageTk.PhotoImage(img)
                
                self.panel.config(image=img)
                self.panel.image = img
                
                # Realizar la predicción
                disease = predict_disease(img_path, model)
                self.result.config(text=f'Enfermedad detectada: {disease}')
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen: {e}")
        else:
            messagebox.showwarning("Advertencia", "No se seleccionó ninguna imagen")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
