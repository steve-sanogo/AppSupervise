# TP4 : Algorithme Constructif Monoplan & N-Parité

Ce dernier TP porte sur l'algorithme incrémental **Monoplan**, conçu pour résoudre des problèmes complexes en construisant dynamiquement l'architecture du réseau.

## Problème : La N-Parité
Le problème de la parité (XOR généralisé à N dimensions) est un problème exhaustif et non linéairement séparable, très difficile pour les réseaux classiques.

## Algorithme Monoplan

Monoplan construit un réseau couche par couche :
1.  Un premier hyperplan tente de classer les exemples.
2.  Si des erreurs persistent, un nouvel hyperplan est ajouté pour corriger spécifiquement les erreurs du précédent.
3.  La cible ($\tau$) est modifiée dynamiquement : $\tau(t+1) = \tau(t) \times \sigma(t)$.

## Fonctionnalités

* **Construction dynamique** : Ajout automatique d'hyperplans jusqu'à séparation complète.
* **Matrice de Poids** : Sauvegarde de la matrice $(N+1) \times (NH+1)$.
* **Visualisation** : Graphique des stabilités pour $N$ variant de 2 à 10.

## Exécution

```bash
python main.py