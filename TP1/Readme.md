# TP1 : Le Perceptron & Données Linéairement Séparables

Ce premier TP se concentre sur l'implémentation native du Perceptron de Rosenblatt et l'étude de sa convergence sur des données synthétiques.

## Objectifs

1.  **Algorithme du Perceptron** : Implémentation des versions **Batch** (mise à jour par époque) et **Online** (mise à jour par exemple).
2.  **Génération de Données (LS)** : Création d'un ensemble de données Linéairement Séparables via un "Perceptron Professeur" ($W^*$) en dimension $N+1$.
3.  **Analyse de Convergence** : Étude du recouvrement ($R$) entre le vecteur poids du professeur et celui de l'élève.

## Fonctionnalités

* Génération de données artificielles (LS) avec biais inclus.
* Calcul du cosinus directeur (Recouvrement $R$) :
    $$R = \frac{W^* \cdot W}{|W^*| \cdot |W|}$$
* Comparaison des performances (Itérations et $R$) en fonction de :
    * La dimension $N$ (2, 10, 100, 1000...)
    * Le nombre d'exemples $P$
    * Le taux d'apprentissage $\eta$ (eta)

## Exécution

```bash
python main.py