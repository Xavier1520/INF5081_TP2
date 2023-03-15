# INF5081 - Travail pratique 2 - Mushroom Wizard

## Auteur
Michael Capone (CAPM17068409)  
Olga Fedorova (FEDO27619308)  
Xavier Dupré (DUPX28029808)

## Installation

Il est nécessaire d'installer `Python 3` sur votre machine pour lancer le script.

Il est nécessaire d'installer les librairies suivantes:
* `Tensorflow 2`
* `Numpy`

Un dossier d'images `Champignons/Champignons` doit se trouver à la racine du projet.

## TP2-train

Ce fichier contient le code nécessaire pour entrainer un CNN. Il génère un fichier nommé `meilleurModel.h5` correspondant au meilleur modèle pour faire des prédictions.

## TP2

Ce fichier contient le code nécessaire pour charger le fichier `meilleurModel.h5` et prédire à quelle classe de champignions appartient l'image donnée en argument.

## meilleurModel.h5

Ce fichier contient les informations nécessaire pour reconstruire le modèle. Il est généré par la méthode `model.save` de Keras.

## imageTricoClasse9

Cette image de champignon appartenant à la classe `Tricholoma scalpturatum`. Elle permet de tester le fichier TP2.py qui requiert le chemin d'une image pour effectuer la prédiction.

## Exécution

Pour lancer l'entrainement du CNN et qui générera le modèle:
```sh
python TP2-train.py
```

Pour tester la prédiction:
```sh
python TP2.py imageTricoClasse9.jpg
```
