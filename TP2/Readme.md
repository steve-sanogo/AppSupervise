# TP2 : Classification Sonar, Pocket & Early Stopping

Ce TP applique les réseaux de neurones à un jeu de données réel et introduit des méthodes pour gérer les données non linéairement séparables.

## Données : Sonar (Mines vs Rocks)

Utilisation du dataset **Sonar** de l'UCI Machine Learning Repository (T. Sejnowski).
* **Entrées** : 60 fréquences (réelles).
* **Sorties** : 2 classes (Mine ou Rocher).
* **Taille** : 208 exemples.

## Algorithmes Implémentés

1.  **Perceptron Simple** : Test de l'apprentissage sur les ensembles Train/Test. Calcul des erreurs $E_a$ (Apprentissage) et $E_g$ (Généralisation).
2.  **Algorithme Pocket** : Variante du Perceptron qui conserve en mémoire les meilleurs poids rencontrés (ceux minimisant l'erreur totale) pour traiter les cas non linéairement séparables.
3.  **Early Stopping** : Division du dataset en 3 (Apprentissage 50%, Validation 20%, Test 30%) pour arrêter l'entraînement avant le sur-apprentissage (overfitting).

## Analyses

* Impact de l'initialisation des poids (Aléatoire vs **Hebb**).
* Analyse de la stabilité des exemples (distance à l'hyperplan).
* Vérification de la non-séparabilité linéaire de l'ensemble complet.

## Exécution

```bash
python main.py