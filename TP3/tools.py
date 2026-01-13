import numpy as np
import pandas as pd
import re

from sklearn.decomposition import PCA
from pathlib import Path

from MinimError import MinimerrorTwoTemp

import matplotlib.pyplot as plt


def data_split(X, y, test_size=0.2, random_state=None, shuffle=True, stratify=None):

    # Convertir en numpy arrays si nécessaire
    X = np.asarray(X)
    y = np.asarray(y)

    n_samples = len(X)

    # Déterminer le nombre d'échantillons de test
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    # Vérifier que test_size est valide
    if n_test == 0 or n_train == 0:
        raise ValueError(f"test_size={test_size} donne n_test={n_test} ou n_train={n_train}")

    # Initialiser le générateur aléatoire
    if random_state is not None:
        np.random.seed(random_state)

    # Créer les indices
    indices = np.arange(n_samples)

    # Split stratifié si demandé
    if stratify is not None:
        stratify = np.asarray(stratify)
        unique_classes, class_counts = np.unique(stratify, return_counts=True)

        train_indices = []
        test_indices = []

        for cls, count in zip(unique_classes, class_counts):
            # Indices pour cette classe
            cls_indices = indices[stratify == cls]

            # Nombre de test pour cette classe (proportionnel)
            n_test_cls = max(1, int(count * test_size))

            # Mélanger les indices de cette classe
            if shuffle:
                np.random.shuffle(cls_indices)

            # Split pour cette classe
            cls_test_indices = cls_indices[:n_test_cls]
            cls_train_indices = cls_indices[n_test_cls:]

            test_indices.extend(cls_test_indices)
            train_indices.extend(cls_train_indices)

        # Convertir en arrays
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        # Mélanger final si demandé (mais en gardant la stratification)
        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)

    else:
        # Split simple (non stratifié)
        if shuffle:
            np.random.shuffle(indices)

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

    # Séparer les données
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def calcul_recouvrement(v, w):
    """
    Calcule le recouvrement R entre deux vecteurs v et w
    selon la formule R = (v . w) / (||v|| * ||w||).

    Args:
        v (list ou np.array): Premier vecteur
        w (list ou np.array): Second vecteur

    Returns:
        float: La valeur du recouvrement R (entre -1 et 1)
    """
    # Conversion en tableaux numpy pour faciliter les calculs
    v = np.array(v)
    w = np.array(w)

    # 1. Calcul du produit scalaire (numérateur)
    produit_scalaire = np.dot(v, w)

    # 2. Calcul des normes (dénominateur)
    norme_v = np.linalg.norm(v)
    norme_w = np.linalg.norm(w)

    # Gestion du cas où un vecteur est nul (division par zéro)
    if norme_v == 0 or norme_w == 0:
        return 0.0

    # 3. Calcul final
    R = produit_scalaire / (norme_v * norme_w)

    return R


def plot_minimerror_pca(model, X, y, title="Minimerror – PCA + hyperplan"):

    # 1. Sécuriser y
    y = np.array(y).reshape(-1).astype(int)

    # 2. Détection des classes
    classes_uniques = np.unique(y)
    if len(classes_uniques) < 2:
        val_neg, val_pos = classes_uniques[0], classes_uniques[0]
    else:
        val_neg, val_pos = classes_uniques[0], classes_uniques[1]

    # 3. Gestion du Scaling
    # On copie X pour ne pas modifier l'original
    X_used = X.copy()

    # Si le modèle a un scaler, on l'utilise, sauf si X a déjà été scalé à l'extérieur
    if model.scale_inputs and model.scaler is not None:
        try:
            # On vérifie si la dimension de X correspond au scaler
            if X.shape[1] == model.scaler.n_features_in_:
                X_used = model.scaler.transform(X)
        except Exception as e:
            print(f"Info Visualisation : Pas de scaling appliqué ({e})")
            pass

    # 4. Calcul PCA
    pca = PCA(n_components=2)
    try:
        X_pca = pca.fit_transform(X_used)
    except Exception as e:
        print(f"ERREUR PCA : Impossible de réduire les dimensions. {e}")
        return

    # --- JITTER ---
    jitter_strength = 0.05
    noise = np.random.normal(0, jitter_strength, size=X_pca.shape)
    X_plot = X_pca + noise

    # 5. Projection de l'Hyperplan
    w_pca = None
    if model.w is not None:
        # On s'assure que w est plat (1D)
        w_flat = model.w.reshape(-1)

        # On sépare w (poids) et b (biais)
        # On suppose que le biais est le DERNIER élément
        w_no_bias = w_flat[:-1]
        b = w_flat[-1]

        # VÉRIFICATION CRITIQUE DES DIMENSIONS
        # PCA.components_ est (2, n_features)
        # w_no_bias doit être (n_features,)
        if pca.components_.shape[1] == w_no_bias.shape[0]:
            # Projection : (2, N) @ (N,) -> (2,)
            w_pca = pca.components_ @ w_no_bias

            # Grille pour la ligne
            x_min, x_max = X_pca[:, 0].min(), X_pca[:, 0].max()
            margin = (x_max - x_min) * 0.1
            x_vals = np.linspace(x_min - margin, x_max + margin, 300)

            if abs(w_pca[1]) < 1e-8:
                y_vals = np.zeros_like(x_vals)
            else:
                y_vals = -(w_pca[0] * x_vals + b) / w_pca[1]
        else:
            print(f"ATTENTION : Impossible de projeter l'hyperplan.")
            print(f"Dim PCA: {pca.components_.shape[1]}, Dim Poids: {w_no_bias.shape[0]}")

    # 6. Affichage
    plt.figure(figsize=(10, 7))

    mask_neg = (y == val_neg)
    mask_pos = (y == val_pos)

    # Bleu
    plt.scatter(
        X_plot[mask_neg, 0], X_plot[mask_neg, 1],
        c='blue', label=f"Classe {val_neg}",
        s=80, alpha=0.6, edgecolor="black"
    )

    # Rouge
    if val_pos != val_neg:
        plt.scatter(
            X_plot[mask_pos, 0], X_plot[mask_pos, 1],
            c='red', label=f"Classe {val_pos}",
            s=80, alpha=0.6, edgecolor="black"
        )

    # Ligne
    if w_pca is not None:
        plt.plot(x_vals, y_vals, "k--", linewidth=2, label="Hyperplan")

    plt.axhline(0, color="gray", linestyle=":", alpha=0.5)
    plt.axvline(0, color="gray", linestyle=":", alpha=0.5)
    plt.title(f"{title} (Avec Jitter)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Limites safe
    y_min, y_max = X_pca[:, 1].min(), X_pca[:, 1].max()
    m_y = (y_max - y_min) * 0.5 if y_max != y_min else 1.0
    plt.ylim(y_min - m_y, y_max + m_y)

    plt.tight_layout()
    plt.show()


def plot_cost_and_derivative(model):
    """
    Fonction universelle : Trace V(γ) et dV/dγ.
    Compatible avec :
      - Minimerror (1 température T)
      - MinimerrorTwoTemp (2 températures β+, β-)
    """

    # --- 1. Détection du type de modèle ---
    mode = "unknown"

    # Cas 1 : Modèle à deux températures (MinimerrorTwoTemp)
    if hasattr(model, 'history') and "beta_plus" in model.history:
        mode = "two_temp"
        b_plus = model.history["beta_plus"][-1]
        b_minus = model.history["beta_minus"][-1]
        # print(f"Mode détecté : TwoTemp (β+={b_plus:.2f}, β-={b_minus:.2f})")

    # Cas 2 : Modèle classique (Minimerror avec T)
    elif hasattr(model, 'history') and "T" in model.history:
        mode = "one_temp"
        T = model.history["T"][-1]
        # Conversion T -> beta pour unifier les formules
        # Si T est très petit, beta devient très grand
        b_plus = 1.0 / T if T > 1e-9 else 1000.0
        b_minus = b_plus  # En mode classique, les deux sont égaux
        # print(f"Mode détecté : Classique (T={T:.4f} -> β={b_plus:.2f})")

    else:
        print("Erreur : Type de modèle non reconnu ou historique vide.")
        return

    # --- 2. Création de l'axe des stabilités (γ) ---
    gamma = np.linspace(-2, 2, 500)

    # --- 3. Calculs Unifiés ---

    # On détermine quel beta utiliser pour chaque point gamma
    if mode == "two_temp":
        # Logique asymétrique : β+ si γ>0, sinon β-
        beta_vals = np.where(gamma > 0, b_plus, b_minus)
    else:
        # Logique symétrique : β est constant partout
        beta_vals = b_plus

        # Formules mathématiques (valables pour les deux cas)
    # Argument de la tangente hyperbolique : z = γ * β / 2
    # Note : pour le modèle classique, γ/2T revient exactement à γ*β/2
    z = gamma * beta_vals / 2
    tanh_z = np.tanh(z)

    # Coût : V = 0.5 * (1 - tanh(z))
    V = 0.5 * (1 - tanh_z)

    # Dérivée : dV = - (β / 4) * (1 - tanh²(z))
    dV = -(beta_vals / 4) * (1 - tanh_z ** 2)

    # --- 4. Affichage ---
    plt.figure(figsize=(9, 6))

    plt.plot(gamma, V, label="V(γ) — Coût", color="black", linewidth=2)
    plt.plot(gamma, dV, label="dV/dγ — Dérivée", color="red", linestyle="--", linewidth=2)

    # Décorations
    plt.axvline(0, color="gray", linestyle=":", alpha=0.5)
    plt.axhline(0, color="gray", linestyle=":", alpha=0.5)

    if mode == "two_temp":
        plt.title(f"Coût Asymétrique (β+={b_plus:.1f}, β-={b_minus:.1f})")
        # Indications visuelles
        plt.text(-1, max(V) * 0.8, f"Erreur (β-)", color='blue', ha='center', alpha=0.6)
        plt.text(1, max(V) * 0.8, f"Confiance (β+)", color='green', ha='center', alpha=0.6)
    else:
        plt.title(f"Coût Symétrique Classique (T={1 / b_plus:.4f})")

    plt.xlabel(r"Stabilité $\gamma$")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_pca_geometric(model, X_data, y_data, stabilities, title):
    """Utilise PCA pour projeter en 2D quand N>2."""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_data)

    # Couleurs
    colors = ['blue' if y == 1 else 'red' for y in y_data]

    fig = plt.figure(figsize=(12, 5))

    # Graphique 1: PCA
    ax1 = fig.add_subplot(121)
    ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=80, edgecolors='black', alpha=0.7)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('Projection PCA (2D)')
    ax1.grid(True, alpha=0.3)

    # Graphique 2: Stabilité
    ax2 = fig.add_subplot(122)

    # Calcul correct de la distance brute (w.x + b)
    # Attention: X_data est l'original, pas le PCA
    w_features = model.w[:-1]
    b = model.w[-1]

    raw_scores = X_data @ w_features + b

    ax2.scatter(raw_scores, stabilities, c=colors, s=80, alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='--')  # Stabilité 0
    ax2.axvline(x=0, color='gray', linestyle='--')  # Score 0

    ax2.set_xlabel('Score brut (w.x + b)')
    ax2.set_ylabel('Stabilité (y * Score)')
    ax2.set_title('Analyse de la stabilité')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_stability_geometric(model, X_data, y_data, title="Représentation géométrique"):
    """
    Dispatche vers la bonne fonction de tracé selon la dimension.
    """
    if model.w is None:
        raise ValueError("Perceptron non entraîné.")

    # Calculer les stabilités
    stabilities = model.compute_test_stabilities(X_data, y_data)

    # Normaliser y pour les couleurs
    y_data = np.where(y_data == 0, -1, y_data)

    n_features = X_data.shape[1]

    if n_features == 1:
        _plot_1d_geometric(model, X_data, y_data, stabilities, title)
    elif n_features == 2:
        _plot_2d_geometric(model, X_data, y_data, stabilities, title)
    else:
        print(f"N={n_features} > 2, utilisation de PCA pour la visualisation")
        _plot_pca_geometric(model, X_data, y_data, stabilities, title)


def _plot_1d_geometric(model, X_data, y_data, stabilities, title):
    """Visualisation pour N=1."""
    plt.figure(figsize=(10, 6))

    # Récupération correcte des poids (Biais à la fin)
    w1 = model.w[0]
    b = model.w[-1]

    # Couleurs par classe
    colors = ['blue' if y == 1 else 'red' for y in y_data]

    plt.scatter(X_data[:, 0], np.zeros_like(X_data[:, 0]),
                c=colors, s=100, alpha=0.8, edgecolors='black')

    # Ligne de décision : w1*x + b = 0 => x = -b/w1
    if w1 != 0:
        x_decision = -b / w1
        plt.axvline(x=x_decision, color='green', linestyle='--',
                    linewidth=3, label=f'Frontière: x={x_decision:.2f}')

    # Flèche du vecteur w (représente la direction positive)
    plt.arrow(x_decision, 0, w1, 0, head_width=0.05, head_length=0.1,
              fc='k', ec='k', linewidth=2, label='Vecteur w')

    plt.xlabel('x₁')
    plt.title(f'{title} - 1D')
    plt.yticks([])  # Pas d'axe Y pertinent
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def _plot_2d_geometric(model, X_data, y_data, stabilities, title):
    """Visualisation pour N=2."""
    fig = plt.figure(figsize=(14, 6))

    # --- Sous-graphique 1: Espace 2D ---
    ax1 = fig.add_subplot(121)

    # Couleurs par classe
    colors = ['blue' if y == 1 else 'red' for y in y_data]

    ax1.scatter(X_data[:, 0], X_data[:, 1],
                c=colors, s=100, alpha=0.8, edgecolors='black')

    # Récupération correcte des poids
    w_features = model.w[:-1]  # Les N premiers
    b = model.w[-1]  # Le dernier
    w_norm = np.linalg.norm(w_features)

    # Tracer le vecteur w (depuis l'origine ou le centre des données)
    origin = np.mean(X_data, axis=0)
    if w_norm > 0:
        scale = 1.0  # Facteur d'échelle visuel
        ax1.arrow(origin[0], origin[1], w_features[0] * scale, w_features[1] * scale,
                  head_width=0.2, head_length=0.2, fc='green', ec='green', linewidth=2,
                  label='Vecteur w')

        # Frontière de décision : w1*x + w2*y + b = 0 => y = -(w1*x + b)/w2
        x_min, x_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
        x_vals = np.linspace(x_min, x_max, 100)

        if abs(w_features[1]) > 1e-8:
            y_vals = -(w_features[0] * x_vals + b) / w_features[1]
            ax1.plot(x_vals, y_vals, 'k--', linewidth=2, label='Hyperplan')
        else:
            # Cas vertical w2=0 => x = -b/w1
            ax1.axvline(x=-b / w_features[0], color='k', linestyle='--')

        # Projections des points sur w (illustration géométrique)
        # Pour projeter sur la direction w : P_proj = (P . w / |w|^2) * w
        # Note: ceci projette sur le vecteur passant par l'origine, pas sur l'hyperplan affine
        for x, y in X_data:
            point = np.array([x, y])
            # Projection orthogonale simple pour visualisation
            # (Optionnel : alourdit parfois le graphique)
            pass

    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_title('Espace des caractéristiques')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Sous-graphique 2: Projection 1D sur w ---
    ax2 = fig.add_subplot(122)

    # Calcul de la distance signée à l'hyperplan : (w.x + b) / ||w||
    # C'est proportionnel à la stabilité avant multiplication par y
    if w_norm > 0:
        distances = (X_data @ w_features + b) / w_norm

        # On trace index vs distance
        # On trie pour la lisibilité
        sort_idx = np.argsort(distances)

        for i, idx in enumerate(sort_idx):
            d = distances[idx]
            c = colors[idx]
            marker = 'o' if stabilities[idx] > 0 else 'x'  # x si erreur
            ax2.scatter(d, i, c=c, marker=marker, s=80, alpha=0.7)

        ax2.axvline(x=0, color='black', linestyle='-', linewidth=2, label='Frontière')
        ax2.set_xlabel('Distance signée à l\'hyperplan')
        ax2.set_title('Marge et Séparabilité')
        ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def parse_sonar_file(filename, path):
    """
    Parse les fichiers sonar.names et sonar.mines :
    '*' => train ; absence de '*' => test

    Retourne 4 DataFrames:
      - train_df (60 colonnes)
      - test_df  (60 colonnes)
      - y_train_df (1 colonne 'class')
      - y_test_df  (1 colonne 'class')
    """
    p = Path(path) / filename
    if not p.exists():
        raise FileNotFoundError(f"Fichier introuvable: {p}")

    with open(p, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex généreuse : accepte IDs de type lettres+chiffres
    pattern = r'^\s*(\*?)\s*([A-Za-z]+\d+)\s*:\s*\{([^}]*)\}'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

    if len(matches) == 0:
        raise ValueError(
            "Aucune entrée trouvée. Vérifie que les lignes ressemblent à '*CM123: {0.1, ...}' "
            "et que les accolades '{ }' et les IDs sont présents."
        )

    train_data = {}
    test_data = {}

    for star, sample_id, values_str in matches:
        # Extraire nombres (supporte 1, 0.5, -0.03)
        nums = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', values_str)
        values = [float(v) for v in nums]

        if len(values) != 60:
            raise ValueError(f"{sample_id} a {len(values)} dimensions (attendu 60). Vérifie la ligne.")

        if star == '*':
            train_data[sample_id] = values
        else:
            test_data[sample_id] = values

    # Avant de renommer: vérifier qu’on a trouvé au moins une entrée
    if len(train_data) == 0 and len(test_data) == 0:
        raise ValueError("Aucune donnée parsée. Le fichier ne contient pas de lignes conformes au format attendu.")

    # Construire DataFrames
    cols = [f'Attribute{i}' for i in range(1, 61)]
    train_df = pd.DataFrame.from_dict(train_data, orient='index')
    test_df = pd.DataFrame.from_dict(test_data, orient='index')

    # Renommer les colonnes (seulement si DataFrame non vide)
    if train_df.shape[1] > 0:
        train_df.columns = cols
    if test_df.shape[1] > 0:
        test_df.columns = cols

    train_df.index.name = 'Sample_ID'
    test_df.index.name = 'Sample_ID'

    # Déterminer le label depuis le nom du fichier
    fn_lower = filename.lower()
    if 'mine' in fn_lower:
        label = 'M'
    elif 'rock' in fn_lower:
        label = 'R'
    else:
        raise ValueError("Impossible d'inférer le label depuis le nom du fichier (attendu 'mines' ou 'rocks').")

    y_train_df = pd.DataFrame({'class': [label] * len(train_df)}, index=train_df.index)
    y_test_df = pd.DataFrame({'class': [label] * len(test_df)}, index=test_df.index)

    return train_df, test_df, y_train_df, y_test_df


def plot_stability_geometric_betas(X_train, y_train, X_test, y_test,
                                   beta_values=range(1, 11),
                                   rapport_temperature=10,
                                   cols=5, epochs=1000, train_verbose=False,
                                   train_epochs_per_beta=None):
    """
    Pour chaque beta dans beta_values : entraîne MinimerrorTwoTemp sur (X_train,y_train)
    puis trace, dans une même figure (grille), le scatter score brut vs stabilité pour X_test.
    - cols : nombre de colonnes dans la grille
    - epochs : nombre d'époques par entraînement (par défaut)
    - train_verbose : si True affiche la progression d'entraînement
    """
    from math import ceil

    beta_list = list(beta_values)
    rows = ceil(len(beta_list) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    fig.suptitle(f"Analyse stabilité pour β ∈ [{beta_list[0]}..{beta_list[-1]}] (rapport={rapport_temperature})", fontsize=14)

    for i, beta in enumerate(beta_list):
        r = i // cols
        c = i % cols
        ax = axes[r][c]

        # Créer et entraîner le modèle pour ce beta
        model = MinimerrorTwoTemp(
            beta0=beta,
            rapport_temperature=rapport_temperature,
            learning_rate=0.02,
            init_method="hebb",
            hebb_noise=1e-3,
            normalize_weights=True,
            scale_inputs=False
        )

        # entraîner (utiliser epochs passés en paramètre)
        e = train_epochs_per_beta if train_epochs_per_beta is not None else epochs
        model.train(X_train, y_train, epochs=e, verbose=train_verbose, beta_max=1000)

        # Préparer labels tests pour affichage / stabilities
        try:
            y_test_prepped = model._prepare_labels(y_test)
        except Exception:
            # fallback: mapper 0->-1 si besoin
            y_test_prepped = np.where(np.asarray(y_test) == 0, -1, np.asarray(y_test))

        # calculs
        stabilities = model.compute_stability(X_test, y_test_prepped)
        w_features = model.w[:-1]
        b = model.w[-1]
        raw_scores = X_test @ w_features + b

        colors = ['blue' if y == 1 else 'red' for y in y_test_prepped]

        ax.scatter(raw_scores, stabilities, c=colors, s=20, alpha=0.75, edgecolors='k', linewidth=0.2)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)

        # afficher Ea sur train et Eg sur test
        Ea, Eg = model.compute_errors(X_train, y_train, X_test, y_test)
        ax.set_title(f"β={beta}  Ea={Ea} Eg={Eg}")
        ax.set_xlabel("Score brut (w.x + b)")
        ax.set_ylabel("Stabilité (y * score)")
        ax.grid(True, alpha=0.25)

    # masquer axes vides
    total = rows * cols
    for j in range(len(beta_list), total):
        r = j // cols
        c = j % cols
        axes[r][c].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()