import numpy as np

from MinimError import Minimerror, MinimerrorTwoTemp
from get_data import GenerateLS

from tools import data_split, plot_stability_geometric
from tools import calcul_recouvrement, plot_stability_geometric_betas
from tools import plot_minimerror_pca, plot_cost_and_derivative


# =============================================
# Partie 1 TP : Minimerror avec une température
# =============================================
def part_I(plot=False):

    np.random.seed(0)  # Reproductibilité

    ls = GenerateLS(n=4, p=100)
    X, y = ls.get_ls()
    w_maitre = ls.perceptron_maitre
    X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.2, random_state=3)

    # CONFIGURATION OPTIMALE POUR Ea = 0
    model = Minimerror(
        T=0.5,
        learning_rate=0.001,
        init_method='hebb',
        hebb_noise=0.0,
        normalize_weights=True,
        scale_inputs=False,  # déjà centrées
        momentum=0.0,
        min_lr_ratio=0.001
    )

    # Entraînement avec la NOUVELLE méthode
    model.train(
        X_train, y_train,
        epochs=800,
        anneal=True,
        T_final=0.001,
        gradient_method='all',  # recommandé pour LS
        early_stopping=True,
        verbose=False)

    r = calcul_recouvrement(w_maitre, model.w)
    print("Recouvrement (R): ", r)
    # a) Calculer les erreurs d’apprentissage Ea et de généralisation Eg;
    y_pred_train = model.predict(X_train)  # as-t-on atteint la convergence parfaite ?

    Ea = np.sum(y_pred_train != y_train)

    y_pred_test = model.predict(X_test)
    Eg = np.sum(y_pred_test != y_test)

    print(f"Ea = {Ea}/{len(y_train)}")
    print(f"Eg = {Eg}/{len(y_test)}")

    # b) Afficher les N+1 poids W du perceptron ;
    print("poids w :", model.w)

    # c) Calculer les stabilités des P exemples de « test » (distance a l’hyperplan séparateur avec les poids normés)
    stabilities = model.compute_test_stabilities(X_test, y_test)

    # d) Graphique des stabilités
    plot_stability_geometric(model, X_test, y_test, "Grpahe des stabilités")

    if plot:  # autre afffichages
        try:
            plot_minimerror_pca(model=model, X=X_train, y=y_train, title="Minimerror – PCA + hyperplan")
        except:
            print("L'affichage de l'hyperplan séparateur à échoué")

        try:
            plot_cost_and_derivative(model=model)
        except:
                print("L'affichage a echoué de la fonction cout et de sa dérivée a échoué")

    return model


# ======================================================================
# Partie 2 TP : Minimerror avec deux températures sur un LS
# ======================================================================


def parti_II():

    np.random.seed(0)  # Reproductibilité

    # Génération de données d'exemple
    ls = GenerateLS(n=4, p=100)
    X, y = ls.get_ls()
    w_maitre = ls.perceptron_maitre

    beta_initial = 2.0
    rapport_temperature = 2.0
    np.random.seed(0)

    # Split train/test
    x_train, x_test, y_train, y_test = data_split(X, y, test_size=0.2, random_state=42)

    print("=" * 60)
    print(f"EXPÉRIENCE AVEC β+={beta_initial}, β-={beta_initial / rapport_temperature} ")
    print("=" * 60)

    # Entraînement

    model = MinimerrorTwoTemp(
        beta0=beta_initial,                    # β initial petit
        rapport_temperature=rapport_temperature,       # ρ = β+/β- = 6
        learning_rate=0.001,           # Taux d'apprentissage modéré
        init_method="hebb",           # Initialisation Hebb
        hebb_noise=1e-3,              # Léger bruit pour Hebb
        normalize_weights=True,        # Normalisation des poids
        scale_inputs=False
    )

    model.train(
        x_train, y_train,
        epochs=800,
        beta_max=1000,           # Température maximale
        b=1
    )

    # Calcul du recouvrement
    r = calcul_recouvrement(w_maitre, model.w)
    print("Recouvrement (R): ", r)

    # a Calcul des erreurs
    y_predict = model.predict(x_train)
    Ea = int(np.sum(y_predict != y_train))
    print(f"\nEa (erreur apprentissage) = {Ea:.4f}")

    # b Sauvegarde des poids
    model.save_weights("Partie-2_weights.txt")

    # c Calculer les stabilités des P exemples de « test »
    stabilites_test = model.compute_test_stabilities(x_test, y_test)
    print("Stabilités test:", stabilites_test)
    # d Graphique des stabilités en fonction de β = 1,2,...10

    print("Étude des stabilités en fonction de β avec rapport fixé à 2")

    plot_stability_geometric_betas(
        x_train, y_train, x_test, y_test,
        beta_values=range(1, 11),
        rapport_temperature=rapport_temperature,
        cols=5,
        epochs=800,
        train_verbose=False
    )


# Lancer les tests
# dé-commenter la fonctionnalité à tester
if __name__ == "__main__":
    part_I(plot=True)
    parti_II()





