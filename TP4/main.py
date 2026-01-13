from Monoplan import Monoplan
from get_data import NParityDataset

import numpy as np
import matplotlib.pyplot as plt



# Fonction de plotting intégrée
def plot_monoplan_stability(model, X, y, title="Stabilité Monoplan (Couche Sortie)"):
    """Trace la stabilité des exemples par rapport à l'hyperplan de sortie"""
    stabs = model.compute_stability(X, y)

    if stabs is None:
        print("Impossible de calculer la stabilité.")
        return

    # Pour l'axe X, on veut le Score Brut (w.x + b)
    # Stabilité = y * Score / ||w||  =>  Score ~ Stabilité * y (à une constante près)
    # C'est suffisant pour la visualisation
    scores_approx = stabs * y

    plt.figure(figsize=(10, 6))

    # Couleurs : Bleu=Succès (+1), Rouge=Echec (-1) ou selon la classe
    # Ici on colore selon la vraie classe pour voir la séparation
    colors = ['blue' if label == 1 else 'red' for label in y]

    # Scatter : Index vs Stabilité
    plt.scatter(range(len(stabs)), stabs, c=colors, s=60, alpha=0.7, edgecolors='k')

    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label="Frontière")
    plt.xlabel("Index de l'exemple")
    plt.ylabel("Stabilité (Marge normalisée)")
    plt.title(title)
    plt.legend(["Frontière", "Classe +1 (Bleu)", "Classe -1 (Rouge)"])
    plt.grid(True, alpha=0.3)

    # Zone d'erreur en rouge clair
    ylim = plt.ylim()
    plt.fill_between(range(len(stabs)), ylim[0], 0, color='red', alpha=0.1)
    plt.ylim(ylim)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=== TEST MONOPLAN SUR N-PARITÉ ===")

    # 1. Génération de données
    np.random.seed(0)
    n = 10 
    dataset = NParityDataset(n)
    X, y = dataset.X, dataset.y
    print(f"Dataset : {len(X)} exemples, {n} dimensions")

    # 2. Création et Entraînement
    # H_max ~ N est suffisant théoriquement
    model = Monoplan(N=n, P=len(X), H_max=n+2)

    print("\n--- Début de l'entraînement ---")
    model.train(X, y, verbose=True)

    # 3. Évaluation
    predictions = model.predict(X)
    n_errors = np.sum(predictions != y)

    print("\n" + "=" * 40)
    print(f"RÉSULTATS FINAUX (N={n})")
    print("=" * 40)
    print(f"Erreurs : {n_errors} / {len(y)}")
    print(f"Succès  : {'OUI' if n_errors == 0 else 'NON'}")

    # 4. Affichage des poids
    print("\n--- Poids du Réseau ---")
    weights = model.get_weights()
    for layer, w in weights.items():
        if isinstance(w, np.ndarray):
            print(f"{layer:10s} : {np.array2string(w, precision=3, suppress_small=True)}")
        else:
            print(f"{layer:10s} : {w}")

    # 5. Calcul et Graphique de Stabilité
    stabs = model.compute_stability(X, y)
    if stabs is not None:
        print(f"\nStabilité Moyenne : {np.mean(stabs):.4f}")
        print(f"Stabilité Min     : {np.min(stabs):.4f}")
        print(f"Afffichage des stabilités: {stabs}")

        # Afficher le graphique
        print("Affichage du graphique de stabilité...")
        try:
            plot_monoplan_stability(model, X, y, title=f"Stabilité Monoplan - N={n}")
        except Exception as e:
            print(f"Erreur graphique : {e}")

    # 6. Sauvegarde
    filename = f"Monoplan_Result_N{n}.txt"
    model.save_model(filename)