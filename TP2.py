# Mettre avant l'import de Tensorflow.
import os
import sys
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

classes = [('Armillaria lutea', 'Non'), 
           ('Coprinellus micaceus', 'Non'), 
           ('Fomes formentarius', 'Non'), 
           ('Fomitopsis pinicola', 'Non'), 
           ('Ganoderma pfeifferi', 'Oui'), 
           ('Mycena galericulata', 'Oui'), 
           ('Plateus cervinus', 'Oui'), 
           ('Plicatura crispa', 'Non'), 
           ('Tricholoma scalpturatum', 'Oui'), 
           ('Xerocomellus chrysentron', 'Oui')]

width = 128
height = 128
color_mode = 'rgb'

# Formatter l'image dans le format attendu par le meilleur modele.
def mushroomImagePreprocessor(image_path, color_mode, target_size):
    image = load_img(image_path, color_mode=color_mode, target_size=target_size)
    image = img_to_array(image)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    return image

# Faire la prédiction sur une image recue en parametre.
# Charger le meilleur modele obtenu lors de l'entrainement (meilleurModel.h5).
# Retourne un numpy array contenant la position de la classe. 
# Exemple: [[0,0,1,0,0,0,0,0,0,0]]
def mushroomWizard(image, model_file):
    model = load_model(model_file)
    prediction = model.predict(image)
    return prediction

# Convertir en language humain la prédiction fait par le Wizard des Champignons.
def mushroomOracle(prediction, classes):
    for image in prediction:
        index = np.argmax(image, axis=-1)
        print("Espece : ", classes[index][0])
        comestible = 0
        for i in range(10):
            if classes[i][1] == 'Oui':
                comestible += image[i]
        if comestible < 0.5:
            print("Comestible : Non")
        else:
            print("Comestible : Oui")

image = mushroomImagePreprocessor(sys.argv[1], color_mode=color_mode, target_size=(width, height))
prediction = mushroomWizard(image, 'meilleurModel.h5')
mushroomOracle(prediction, classes)
