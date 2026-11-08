# Desactivo los mensajes de advertencia de TensorFlow
import os
# Oculta los mensajes informativos y de advertencia de C++ de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
# Desactiva los avisos de round-off de oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Importo las librerías necesarias para el modelo
import tensorflow as tf
from tensorflow.keras import layers, models

# Importo la librería para graficar imágenes
import matplotlib.pyplot as plt

# Carga del conjunto de datos MNIST en blanco y negro
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Se normalizacion las intensidades de los pixeles
# 1 pixel de blanco y negro esta entre 0-255
# se divide entre el maximo valor posible y queda entre 0-1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Se agrega un canal para el procesamiento de los colores de las imagenes
# en este caso no hay color entonces la cantidad de canales es 1
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)


# # Esto convierte tu única matriz de 28x28 en la imagen que realmente es
# plt.imshow(x_train[0, :, :, 0], cmap='gray')
# plt.show()


# Construir la arquitectura del modelo
model = models.Sequential([
    # Primera capa convolucional

    # filtros: 32
    # forma del filtro: 3x3
    # funcion de activacion: relu
    # forma de la entrada (solo para la primera capa): (28x28) pixeles x 1 canal 
    layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),

    # Primera capa de Max pooling

    # Se toma el mayor numero de la mascara
    # forma de la mascara: 2x2
    layers.MaxPooling2D((2, 2)),
    # El stride es de 2 por default aunque no se especifique
    
    # Salgo de la primer capa con dimension 14x14x8 (8 filtros)
    
    # Segunda capa convolucional y max pooling
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Tercera capa convolucional
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    
    # Aplanar los datos para las capas densas
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    
    # Capa de salida: 10 neuronas (números del 0 al 9) con función softmax para probabilidades
    layers.Dense(10, activation='softmax')
])

# Compilo el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Mostrar un resumen de la arquitectura
model.summary()

# Entrenamiendo del modelo el modelo
print("\nIniciando el entrenamiento...")
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Evaluo con los datos de prueba
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"\nPrecisión en los datos de prueba: {test_acc * 100:.2f}%")

