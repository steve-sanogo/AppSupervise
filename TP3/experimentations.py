import numpy as np
import pandas as pd

from tools import plot_minimerror_pca, plot_cost_and_derivative
from tools import parse_sonar_file

from MinimError import Minimerror, MinimerrorTwoTemp
from get_data import NParityDataset, theoretical_min_errors


# ===============================================
# Minimerror avec une température sur la N-Parité
# ===============================================

def n_parity_test(n, plot=False):
    np.random.seed(0)

    dataset = NParityDataset(n)
    X, y = dataset.X, dataset.y

    print(f"\n--- Test N-Parité (1 Temp) pour N={n} ---")

    target_errors = theoretical_min_errors(n)
    print(f"Objectif théorique : {target_errors} erreurs")

    model = Minimerror(
        T=5,
        learning_rate=0.06,
        init_method='hebb',
        hebb_noise=0.001,
        normalize_weights=True,
        scale_inputs=True,
        momentum=0,
        min_lr_ratio=0.001
    )

    model.train(
        X=X, y=y,
        epochs=300,
        anneal=True,
        T_final=0.05,
        gradient_method='all',
        early_stopping=False,
        verbose=False
    )

    y_pred = model.predict(X)
    Ea = np.sum(y_pred != y)
    success = (Ea == target_errors)

    print(f"Résultat : {Ea} erreurs (Succès: {'OUI' if success else 'NON'})")

    if plot:
        try:
            plot_minimerror_pca(model=model, X=X, y=y, title=f"Minimerror N-{n} Parité")
        except Exception as e:
            print(f"Erreur Plot PCA: {e}")

        try:
            plot_cost_and_derivative(model=model)
        except Exception as e:
            print(f"Erreur Plot Coût: {e}")

    return model


# =================================================
# Minimerror avec deux températures sur la N-Parité
# =================================================

def parity_with_two_temp(n, plot=False):
    np.random.seed(0)
    dataset = NParityDataset(n)
    X, y = dataset.X, dataset.y

    print(f"\n--- Test N-Parité (2 Temp) pour N={n} ---")

    # Hyperparamètres dynamiques
    if n < 8:
        epochs_n = 500
        noise = 0.001
        decay = 0.99
    elif n == 8:
        epochs_n = 5000
        noise = 0.005
        decay = 0.9995
    else:
        # N >= 9
        epochs_n = 25000
        noise = 0.008
        decay = 0.9999

    target_errors = theoretical_min_errors(n)
    print(f"Objectif théorique : {target_errors} erreurs")

    model = MinimerrorTwoTemp(
        beta0=0.2,
        rapport_temperature=6,
        learning_rate=0.02,
        init_method="hebb",
        normalize_weights=True,
        hebb_noise=noise,
        scale_inputs=True,
        min_lr_ratio=0.0001
    )

    model.train(X, y,
                epochs=epochs_n,
                beta_max=100,
                b=decay,
                early_stopping=False,
                verbose=False
                )

    y_pred = model.predict(X)
    Ea = np.sum(y_pred != y)
    success = (Ea == target_errors)

    print(f"Résultat : {Ea} erreurs (Succès: {'OUI' if success else 'NON'})")

    if plot:
        try:
            plot_minimerror_pca(model=model, X=X, y=y, title=f"Minimerror 2T N-{n} Parité")
        except Exception as e:
            print(f"Erreur Plot PCA: {e}")

        try:
            plot_cost_and_derivative(model=model)
        except Exception as e:
            print(f"Erreur Plot Coût: {e}")

    return model


# ===============================================================
# Minimerror avec avec Ea=0 pour le Sonar avec deux temperatures
# ==============================================================

file_path = r"../data"
# Initialisation globale
x_train, y_train = None, None
x_test, y_test = None, None


def load_sonar_data():
    global x_train, y_train, x_test, y_test

    # On suppose que parse_sonar_file est dans tools et retourne des DataFrames
    try:
        xt_m, xv_m, yt_m, yv_m = parse_sonar_file("sonar.mines", file_path)
        xt_r, xv_r, yt_r, yv_r = parse_sonar_file("sonar.rocks", file_path)
    except Exception as e:
        print(f"Erreur chargement données: {e}")
        return

    # Concaténation Pandas
    x_train_df = pd.concat([xt_m, xt_r])
    x_test_df = pd.concat([xv_m, xv_r])
    y_train_df = pd.concat([yt_m, yt_r])
    y_test_df = pd.concat([yv_m, yv_r])

    # Mapping M/R -> 1/-1
    y_train_df = y_train_df.copy()
    y_test_df = y_test_df.copy()

    y_train_df['class'] = y_train_df['class'].map({'M': 1, 'R': -1})
    y_test_df['class'] = y_test_df['class'].map({'M': 1, 'R': -1})

    # Conversion Numpy finale
    x_train = x_train_df.to_numpy()
    x_test = x_test_df.to_numpy()
    y_train = y_train_df['class'].to_numpy()
    y_test = y_test_df['class'].to_numpy()

    print(f"Données Sonar chargées : Train {x_train.shape}, Test {x_test.shape}")


def zero_ea_with_sonar(plot=False):

    np.random.seed(5)  # la valeur de seed varie selon les machines pour la convergence
    load_sonar_data()

    if x_train is None:
        print("Annulation : Données non chargées.")
        return

    print("\n--- Test Sonar (Objectif Ea=0) ---")

    model = MinimerrorTwoTemp(
        beta0=0.2,
        rapport_temperature=15,
        learning_rate=0.02,
        init_method="hebb",
        hebb_noise=0.01,
        normalize_weights=True,
        scale_inputs=True,
    )

    model.train(x_train, y_train,
                epochs=3000,
                early_stopping=True,
                verbose=True,
                beta_max=120,
                b=0.995)

    y_pred = model.predict(x_train)
    Ea = np.sum(y_pred != y_train)
    print(f"Ea (Apprentissage) = {Ea} / {len(y_train)}")

    y_pred_test = model.predict(x_test)
    Eg = np.sum(y_pred_test != y_test)
    print(f"Eg (Généralisation) = {Eg} / {len(y_test)}")

    # Sauvegarde
    filename = f"Sonar_weights_Ea={Ea}_2temps.txt"
    model.save_weights(filename)

    if plot:
        try:
            plot_minimerror_pca(model=model, X=x_train, y=y_train, title="Minimerror Sonar")
        except Exception as e:
            print(f"Erreur PCA Sonar: {e}")

        try:
            plot_cost_and_derivative(model=model)
        except Exception as e:
            print(f"Erreur Coût Sonar: {e}")

# ===============================================================
# Minimerror avec avec Ea=0 pour le Sonar avec une temperature
# Selection intelligente des données pour le calcul de w
# ==============================================================


def zero_ea_with_gradient_intelligent():

    np.random.seed(5)
    load_sonar_data()

    if x_train is None:
        print("Annulation : Données non chargées.")
        return

    # Entraînement perceptron sur l'ensemble L
    model = Minimerror(
        T=100,  # Température initiale
        learning_rate=0.002,  # Learning rate adaptatif
        init_method='hebb',  # Meilleure initialisation
        hebb_noise=0,  # Léger bruit
        normalize_weights=True,
        scale_inputs=True,  # Important
        momentum=0.98,  # Accélération
        min_lr_ratio=0.0001  # LR ne va pas à 0
    )

    # Entraînement avec la NOUVELLE méthode
    model.train(
        x_train, y_train,
        epochs=5000,  # Suffisant
        anneal=True,
        T_final=0.05,  # Température finale basse
        gradient_method='auto',  # NOUVEAU : sélection intelligente des données pour le calcul du poids
        early_stopping=True,  # Arrête si Ea = 0
        verbose=True
    )

    y_pred_train = model.predict(x_train)
    Ea = np.sum(y_pred_train != y_train)

    if Ea > 0:
        print("\nForçage de la convergence parfaite...")
        model.force_perfect_separation(x_train, y_train)

    y_pred_train = model.predict(x_train)
    Ea = np.sum(y_pred_train != y_train)

    y_pred_test = model.predict(x_test)
    Eg = np.sum(y_pred_test != y_test)

    print(f"Ea = {Ea}/{len(y_train)}")
    print(f"Eg = {Eg}/{len(y_test)}")

    print("\n Stabilité moyenne : ", np.mean(model.compute_test_stabilities(X_test=x_test, y_test=y_test)))


# Lancer les tests
# dé-commenter la fonctionnalité à tester
if __name__ == "__main__":
    # la N-Parité
    for i in range(1, 11):
        n_parity_test(i, plot=True)  # avec une temperature

    for i in range(1, 11):
        parity_with_two_temp(i, plot=True)  # avec deux temperatures

    # SONAR
    # zero_ea_with_sonar(plot=True)  # avec deux températures
    # zero_ea_with_gradient_intelligent() # avec une temperature selection intelligente des points pour w
