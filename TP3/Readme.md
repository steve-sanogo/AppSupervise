# TP3 : Algorithme Minimerror

Implémentation de l'algorithme **Minimerror** (Torres & Gordon), une méthode d'optimisation avancée basée sur une fonction de coût paramétrée par une température $T$.

## Théorie

Contrairement au Perceptron classique, Minimerror minimise une fonction de coût via une descente de gradient et un **recuit déterministe**.
* **Fonction de coût** : Utilise une tangente hyperbolique ($\tanh$) pour lisser l'erreur.
* **Température ($T$)** : Contrôle la fenêtre d'influence des exemples sur la modification des poids.

## Implémentation

### Partie I : Minimerror Standard
* Apprentissage avec une température unique $\beta = 1/T$.
* Calcul des stabilités et visualisation graphique.

### Partie II : Minimerror à 2 Températures
* Introduction de deux températures ($\beta+$ et $\beta-$) pour gérer différemment les exemples bien classés et mal classés.
* Étude de la robustesse via le graphique des stabilités en fonction de $\beta$.

## Exécution

```bash
python main.py