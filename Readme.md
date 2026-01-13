# Apprentissage Supervisé - Implémentation d'Algorithmes Neuronaux

Ce dépôt regroupe l'ensemble des Travaux Pratiques (TP) réalisés dans le cadre du module d'**Apprentissage Supervisé** du Master 1 Intelligence Artificielle.

L'objectif principal est l'implémentation *from scratch* (sans librairies de Deep Learning comme PyTorch, TensorFlow ou Scikit-Learn pour les modèles) d'algorithmes fondamentaux et avancés de réseaux de neurones.

**Université :** Avignon Université - CERI  
**Professeur :** Juan-Manuel Torres Moreno  
**Année :** 2025

## Contenu du dépôt

Le projet est divisé en 4 étapes progressives :

* **[TP1 - Le Perceptron](./TP1)** : Implémentation de l'algorithme du Perceptron (versions Batch et Online) et validation sur des données linéairement séparables (LS) générées artificiellement (modèle Professeur/Élève).
* **[TP2 - Sonar & Pocket](./TP2)** : Application sur le jeu de données réel *Sonar* (UCI). Implémentation de l'algorithme **Pocket** et de la technique d'**Early Stopping** pour gérer la généralisation.
* **[TP3 - Minimerror](./TP3)** : Implémentation de l'algorithme **Minimerror** (optimisation par recuit déterministe) avec gestion d'une et deux températures ($\beta$).
* **[TP4 - Monoplan](./TP4)** : Implémentation de l'algorithme constructif/incrémental **Monoplan** pour résoudre le problème complexe de la N-Parité.

## Structure des dossiers

```text
AppSupervise/
│
├── data/               # Jeux de données (Sonar, etc.)
├── docs/               # Énoncés et supports de cours
├── TP1/                # Perceptron & Données LS
├── TP2/                # Pocket, Early Stopping & Sonar
├── TP3/                # Minimerror (1 & 2 Températures)
├── TP4/                # Monoplan & N-Parité
├── requirements.txt    # Dépendances Python
└── README.md           # Ce fichier