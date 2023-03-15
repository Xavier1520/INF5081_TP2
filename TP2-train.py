# Mettre avant l'import de Tensorflow.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

classes = ['Armillaria lutea', 
           'Coprinellus micaceus', 
           'Fomes formentarius', 
           'Fomitopsis pinicola', 
           'Ganoderma pfeifferi', 
           'Mycena galericulata', 
           'Plateus cervinus', 
           'Plicatura crispa', 
           'Tricholoma scalpturatum', 
           'Xerocomellus chrysentron']

# Parametres et hyperparametres
width = 128
height = 128
color_mode = 'rgb'
channels = 3
learning_rate = 0.0001
batch_size = 16
epochs = 100

# Chemin vers le dossier contenant les sous-repertoires des images.
dataset_path = './Champignons/Champignons'

# ImageDataGenerator de Keras pour augmenter les images en temps réel.
# Reserve 20% du dataset pour la validation.
# Rescale de 1/255 pour normaliser les pixels RGB entre 0 et 1.
# Parametres pour l'augmentation des images 
datagen = ImageDataGenerator(validation_split=0.2,
                             rescale=1./255,
                             rotation_range=45,
                             brightness_range=[0.75, 1.25],
                             shear_range=0.25,
                             zoom_range=0.5,
                             horizontal_flip=True,
                             vertical_flip=True)

train_batches = datagen.flow_from_directory(dataset_path, 
                                            target_size=(width, height),
                                            batch_size=batch_size,
                                            color_mode=color_mode,
                                            class_mode='categorical',
                                            classes=classes,
                                            shuffle='True',
                                            seed=42, 
                                            subset='training')

valid_batches = datagen.flow_from_directory(dataset_path,
                                            target_size=(width, height),
                                            batch_size=batch_size,
                                            color_mode=color_mode,
                                            class_mode='categorical',
                                            classes=classes,
                                            shuffle='True',
                                            seed=42, 
                                            subset='validation')

# Construction du CNN
# Couche INPUT est la premiere couche ajoutee au modele.
# Derniere couche doit avoir autant d'unites que de classes.
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(width, height, channels)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(units=10, activation='softmax'))

# Imprime les information du modèle à la console.
model.summary()

# Optimiseur Adam pour la descente du gradient.
# Fonction de perte Categorical cross-entropy.
# Accuracy a chaque batch.
model.compile(optimizer=Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), 
              loss=CategoricalCrossentropy(), 
              metrics=['accuracy', 'Precision', 'Recall'])

train_size = len(train_batches)
valid_size = len(valid_batches)
model_checkpoint = ModelCheckpoint(filepath='meilleurModel.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

model.fit(train_batches,
          steps_per_epoch=train_size, 
          validation_data=valid_batches,
          validation_steps=valid_size, 
          epochs=epochs, 
          verbose=2,
          callbacks=[model_checkpoint])
