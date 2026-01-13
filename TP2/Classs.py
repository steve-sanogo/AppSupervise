import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # Réduction de dimention, utilisé pour les plots
from numpy import linalg as LA


class Perceptron:
    def __init__(self, eta=0.01, max_epoch=500, pocket=False, pocket_threshold=0, early_stopping=False):
        """
        Initialise un perceptron avec différentes options d'apprentissage.

        Parameters:
        -----------
        eta : float
            Taux d'apprentissage (learning rate)
        max_epoch : int
            Nombre maximum d'époques d'apprentissage
        pocket : bool
            Active l'algorithme Pocket (conserve les meilleurs poids)
        pocket_threshold : int
            Seuil d'erreur pour arrêter Pocket plus tôt
        early_stopping : bool
            Active l'arrêt précoce basé sur l'erreur de validation
        """
        self.eta = eta  # learning rate
        self.max_epoch = max_epoch
        self.w = None  # Poids courants
        self.w_pocket = None  # Meilleurs poids trouvés (Pocket)
        self.errors_history = []  # Historique des erreurs par époque
        self.best_errors_history = []  # Historique des meilleures erreurs
        self.min_error = float('inf')  # Meilleure erreur rencontrée
        self.current_error = None  # Erreur courante
        self.pocket = pocket
        self.pocket_threshold = pocket_threshold
        self.stabilities = None  # Stabilités calculées
        self.epoch = 0  # Nombre d'époques effectuées
        self.early_stopping = early_stopping
        self.error_validation_history = []  # Historique des erreurs de validation
        self._assert_modes_mutually_exclusive()

    def _assert_modes_mutually_exclusive(self):
        """Vérifie que les modes Pocket et early_stopping ne sont pas activés simultanément."""
        if self.pocket and self.early_stopping:
            raise ValueError("Les modes 'pocket' et 'early_stopping' sont mutuellement exclusifs.")

    @staticmethod
    def sign(h):
        """Fonction de seuil."""
        return 1 if h >= 0 else 0

    def initialize_perceptron_vectorized(self, x_train, y_train):
        """
        Version vectorisée de l'initialisation Hebb
        w_j = Σ_{μ=1}^p x_j^μ τ^μ = X^T · τ

        Parameters:
        -----------
        x_train : array (n_samples, n_features)
            Données d'entraînement
        y_train : array (n_samples,)
            Labels d'entraînement {0, 1}

        Returns:
        --------
        bool : True si initialisation réussie
        """
        # Convertir y en {-1, 1}
        tau = 2 * y_train.flatten() - 1  # τ^μ ∈ {-1, 1}

        # Calcul vectorisé pour les features: w_features = X^T · τ
        w_features = np.dot(x_train.T, tau)

        # Normalisation optionnelle (évite les poids trop grands)
        norm = np.linalg.norm(w_features)
        if norm > 0:
            w_features = w_features / norm

        # Initialisation du biais
        w_bias = 0.0

        self.w = np.concatenate([[w_bias], w_features])
        return True

    def compute_errors(self, X, y, w):
        """
        Calcule le nombre d'erreurs avec les poids w.

        Parameters:
        -----------
        X : array (n_samples, n_features+1)
            Données AVEC biais
        y : array (n_samples,)
            Labels vrais
        w : array (n_features+1,)
            Vecteur de poids

        Returns:
        --------
        int : Nombre d'erreurs
        """
        # Version vectorisée pour la performance
        predictions = np.dot(X, w)
        y_pred = np.where(predictions >= 0, 1, 0)
        return np.sum(y_pred != y)

    def train_online(self, x_train, y_train, x_val=None, y_val=None, init_method="random", shuffle=True):
        """
        Algorithme du perceptron en ligne (stochastique) avec option Pocket.

        Parameters:
        -----------
        x_train : array (n_samples, n_features)
            Données d'entraînement
        y_train : array (n_samples,)
            Labels d'entraînement {0, 1}
        x_val : array (n_val_samples, n_features), optional
            Données de validation pour early_stopping
        y_val : array (n_val_samples,), optional
            Labels de validation pour early_stopping
        init_method : str
            Méthode d'initialisation: "random" ou "Hebb"
        shuffle : bool
            Mélanger les données à chaque époque

        Returns:
        --------
        int : Erreur finale (min_error si Pocket, current_error sinon)
        """
        if init_method not in ["random", "Hebb"]:
            raise ValueError("Les valeurs autorisées sont : 'random', 'Hebb'")

        # Vérification des données de validation si early_stopping activé
        if self.early_stopping:
            if x_val is None or y_val is None:
                raise ValueError("Données de validation requises pour l'arrêt précoce.")
            # Préparer les données de validation avec biais
            X_val = np.c_[np.ones(x_val.shape[0]), x_val]
            y_val_flat = y_val.flatten()
            best_val_error = float('inf')
            best_w = None

        # Ajout du biais aux données d'entraînement
        X = np.c_[np.ones(x_train.shape[0]), x_train]
        y = y_train.reshape(-1)

        # Initialisation des poids
        if init_method == 'Hebb':
            self.initialize_perceptron_vectorized(x_train, y_train)
        else:
            # Initialisation aléatoire selon une distribution normale standard
            self.w = np.random.randn(X.shape[1])

        # Initialisation selon le mode (Pocket ou standard)
        current_errors = self.compute_errors(X, y, self.w)

        if self.pocket:
            self.w_pocket = self.w.copy()
            self.min_error = current_errors
            print(f"Erreur initiale Pocket: {self.min_error}/{len(y)}")
        else:
            self.current_error = current_errors
            print(f"Erreur initiale: {self.current_error}/{len(y)}")

        print(f"=========== Début apprentissage Online (Pocket={'activé' if self.pocket else 'désactivé'}) ===========")

        # Initialisation pour la boucle d'apprentissage
        stop_reason = None

        for epoch in range(self.max_epoch):
            errors = 0
            indices = np.arange(len(X))

            if shuffle:
                indices = np.random.permutation(len(X))

            # Une époque d'apprentissage en ligne
            for i in indices:
                h = np.dot(X[i], self.w)
                y_pred = self.sign(h)

                if y_pred != y[i]:
                    self.w += self.eta * (y[i] - y_pred) * X[i]
                    errors += 1

            # Calcul de l'erreur APRÈS l'époque complète
            current_errors = self.compute_errors(X, y, self.w)

            # Mise à jour selon le mode
            if self.pocket:
                if current_errors < self.min_error:
                    improvement = self.min_error - current_errors
                    self.min_error = current_errors
                    self.w_pocket = self.w.copy()
                    if improvement > 0:
                        print(f"Époque {epoch + 1:3d}: Pocket amélioré ({current_errors} erreurs)")

            if self.early_stopping:
                # Calcul de l'erreur sur l'ensemble de validation
                val_errors = self.compute_errors(X_val, y_val_flat, self.w)
                self.error_validation_history.append(val_errors)

                if val_errors < best_val_error:
                    best_val_error = val_errors
                    best_w = self.w.copy()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                # Arrêt si pas d'amélioration pendant plusieurs époques
                if epochs_without_improvement >= 5:  # Patience de 5 époques
                    stop_reason = f"Arrêt précoce à l'époque {epoch + 1} (pas d'amélioration sur validation)"
                    if best_w is not None:
                        self.w = best_w
                    break
            else:
                self.current_error = current_errors

            # Historique des erreurs
            self.errors_history.append(errors)

            if self.pocket:
                self.best_errors_history.append(self.min_error)
            else:
                self.best_errors_history.append(current_errors)

            # Vérification des critères d'arrêt
            if self.pocket and self.min_error <= self.pocket_threshold:
                stop_reason = f"Critère Pocket atteint (erreur ≤ {self.pocket_threshold})"
            elif current_errors == 0:
                stop_reason = "Séparation linéaire parfaite atteinte"
                if self.pocket:
                    self.w_pocket = self.w.copy()
                    self.min_error = 0
            elif epoch == self.max_epoch - 1:
                stop_reason = f"Nombre maximum d'époques atteint ({self.max_epoch})"

            # Arrêt si critère rempli
            if stop_reason:
                self.epoch = epoch + 1
                print(f"\n=========== {stop_reason} ===========")
                print(f"Dernière époque: {epoch + 1}")
                if self.pocket:
                    print(f"Meilleure erreur Pocket: {self.min_error}/{len(y)}")
                else:
                    print(f"Erreur finale: {current_errors}/{len(y)}")
                break

        # Si on sort sans break explicite (normalement non atteint)
        if stop_reason is None:
            self.epoch = self.max_epoch
            print(f"\n=========== Arrêt après {self.max_epoch} époques ===========")
            if self.pocket:
                print(f"Meilleure erreur Pocket: {self.min_error}/{len(y)}")
            else:
                print(f"Erreur finale: {current_errors}/{len(y)}")

        # Utiliser les meilleurs poids si Pocket activé
        if self.pocket:
            self.w = self.w_pocket.copy()
            print(f"Utilisation des poids Pocket (erreur: {self.min_error}/{len(y)})")
        elif self.early_stopping and best_w is not None:
            self.w = best_w
            self.current_error = best_val_error
        else:
            self.current_error = current_errors

        print("=========== Apprentissage Online terminé ===========")

        return self.min_error if self.pocket else self.current_error

    def train_with_batch(self, x_train, y_train, init_method="random", shuffle=True):
        """
        Algorithme du perceptron par batch (non implémenté dans cette version).
        Pour des raisons pédagogiques, nous utilisons train_online() à la place.
        """
        print("Note: train_with_batch() n'est pas implémenté. Utilisation de train_online()...")
        return self.train_online(x_train, y_train, init_method=init_method, shuffle=shuffle)

    def get_perceptron(self):
        """Retourne une copie des poids du perceptron."""
        return self.w.copy() if self.w is not None else None

    def predict(self, x_test):
        """
        Prédit les classes pour un ensemble de test.

        Parameters:
        -----------
        x_test : array (n_samples, n_features)
            Données de test SANS biais

        Returns:
        --------
        array : Prédictions {0, 1}
        """
        if self.w is None:
            raise ValueError("Perceptron non entraîné. Appelez train_* d'abord.")

        # Vérifier la dimension
        if x_test.shape[1] != len(self.w) - 1:
            raise ValueError(f"Dimension incorrecte: attendu {len(self.w) - 1} features, reçu {x_test.shape[1]}")

        # Ajout du biais
        x_test_with_bias = np.c_[np.ones(x_test.shape[0]), x_test]

        # Calcul des produits scalaires
        h = np.dot(x_test_with_bias, self.w)

        # Application de la fonction seuil
        return np.where(h >= 0, 1, 0).reshape(-1, 1)

    def get_ea(self):
        """
        Retourne l'erreur d'apprentissage.

        Returns:
        --------
        int or None : Erreur d'apprentissage
        """
        if self.pocket:
            if self.min_error == float('inf'):
                return None
            return self.min_error
        else:
            if self.current_error is None:
                # Si current_error pas défini, prendre la dernière de l'historique
                if self.errors_history:
                    return self.errors_history[-1]
                return None
            return self.current_error

    def get_eg(self, X_test, y_true):
        """
        Calcule l'erreur en généralisation.

        Parameters:
        -----------
        X_test : array (n_samples, n_features)
            Données de test SANS biais
        y_true : array (n_samples,)
            Vraies classes

        Returns:
        --------
        int : Nombre d'erreurs
        """
        y_pred = self.predict(X_test)
        # Nombre d'erreurs
        errors = np.sum(y_pred.flatten() != y_true.flatten())
        return errors

    def compute_stability(self, X_data, y_data):
        """
        Calcule la stabilité (marge normalisée) pour chaque exemple.
        Formule : γ^μ = (τ^μ * (w·x^μ)) / ||w||
        où τ^μ ∈ {-1, 1} mais nos classes sont {0, 1}

        Parameters:
        -----------
        X_data : array (n_samples, n_features)
            Données SANS biais
        y_data : array (n_samples,)
            Classes {0, 1}

        Returns:
        --------
        array : Stabilités pour chaque exemple
        """
        if self.w is None:
            raise ValueError("Perceptron non entraîné.")

        # Convertir y de {0, 1} à {-1, 1} pour la formule
        y_bipolar = 2 * y_data.flatten() - 1  # 0→-1, 1→+1

        # Ajouter biais aux données
        X_with_bias = np.c_[np.ones(X_data.shape[0]), X_data]

        # Produits scalaires w·x^μ
        h = np.dot(X_with_bias, self.w)

        # Calcul des stabilités : γ^μ = (τ^μ * h^μ) / ||w||
        norm_w = LA.norm(self.w, ord=2)  # Norme euclidienne
        if norm_w == 0:
            raise ValueError("Norme du vecteur poids nulle.")

        self.stabilities = (y_bipolar * h) / norm_w

        return self.stabilities.copy()

    def plot_stability_geometric(self, X_data, y_data, title="Représentation géométrique des stabilités"):
        """
        Représente géométriquement le perceptron (vecteur w) et chaque point
        selon sa distance/projection sur w.

        Parameters:
        -----------
        X_data : array (n_samples, n_features)
            Données
        y_data : array (n_samples,)
            Labels
        title : str
            Titre du graphique
        """
        if self.w is None:
            raise ValueError("Perceptron non entraîné.")

        # Calculer les stabilités
        stabilities = self.compute_stability(X_data, y_data)

        n_features = X_data.shape[1]

        if n_features == 2:
            self._plot_2d_geometric(X_data, y_data, stabilities, title)
        elif n_features == 1:
            self._plot_1d_geometric(X_data, y_data, stabilities, title)
        else:
            print(f"N={n_features} > 2, utilisation de PCA pour la visualisation")
            self._plot_pca_geometric(X_data, y_data, stabilities, title)

    def _plot_1d_geometric(self, X_data, y_data, stabilities, title):
        """Visualisation pour N=1 (1 feature)."""
        plt.figure(figsize=(10, 6))

        # Points colorés par stabilité
        scatter = plt.scatter(X_data[:, 0], np.zeros_like(X_data[:, 0]),
                              c=stabilities, cmap='coolwarm',
                              s=100, alpha=0.8, edgecolors='black')

        # Ligne du perceptron (droite verticale pour w0 + w1*x = 0)
        w0, w1 = self.w[0], self.w[1]
        if w1 != 0:
            x_decision = -w0 / w1
            plt.axvline(x=x_decision, color='red', linestyle='--',
                        linewidth=3, label=f'Frontière: x={x_decision:.2f}')

        # Flèche du vecteur w (projeté sur x)
        plt.arrow(0, -0.5, w1, 0, head_width=0.1, head_length=0.1,
                  fc='green', ec='green', linewidth=3,
                  label=f'w₁={w1:.2f} (composante x)')

        plt.colorbar(scatter, label='Stabilité γ')
        plt.xlabel('x₁')
        plt.title(f'{title} - N=1')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([-1, 1])
        plt.show()

    def _plot_2d_geometric(self, X_data, y_data, stabilities, title):
        """Visualisation pour N=2 (2 features)."""
        fig = plt.figure(figsize=(14, 6))

        # Sous-graphique 1: Points et vecteur w
        ax1 = fig.add_subplot(121)

        # Points colorés par stabilité
        scatter = ax1.scatter(X_data[:, 0], X_data[:, 1],
                              c=stabilities, cmap='coolwarm',
                              s=100, alpha=0.8, edgecolors='black')

        # Vecteur w (sans le biais w0 pour la visualisation)
        w_features = self.w[1:]  # w1 et w2
        w_norm = np.linalg.norm(w_features)

        if w_norm > 0:
            # Normaliser pour une flèche de taille raisonnable
            scale = 2.0 / w_norm if w_norm > 0 else 1
            w_scaled = w_features * scale

            # Flèche du vecteur w
            ax1.arrow(0, 0, w_scaled[0], w_scaled[1],
                      head_width=0.2, head_length=0.3,
                      fc='green', ec='green', linewidth=3,
                      label=f'w=[{w_features[0]:.2f}, {w_features[1]:.2f}]')

            # Ligne perpendiculaire à w (frontière de décision)
            # w0 + w1*x + w2*y = 0 -> y = -(w0 + w1*x)/w2
            w0, w1, w2 = self.w[0], self.w[1], self.w[2]
            x_min, x_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
            x_vals = np.linspace(x_min, x_max, 100)

            if abs(w2) > 1e-10:
                y_vals = -(w0 + w1 * x_vals) / w2
                ax1.plot(x_vals, y_vals, 'r--', linewidth=2,
                         label='Frontière w·x=0')

            # Projection des points sur w (ligne pointillée)
            for i, (x, y) in enumerate(X_data):
                point = np.array([x, y])
                # Projection sur w: (point·w) * w / ||w||²
                if w_norm > 0:
                    proj = np.dot(point, w_features) / (w_norm ** 2) * w_features
                    ax1.plot([x, proj[0]], [y, proj[1]], 'gray',
                             alpha=0.3, linewidth=0.5)

        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        ax1.set_title('Points et vecteur w')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Stabilité γ')

        # Sous-graphique 2: Distribution des projections sur w
        ax2 = fig.add_subplot(122)

        # Calculer les projections (distances signées)
        if w_norm > 0:
            projections = []
            for point in X_data:
                proj_dist = np.dot(point, w_features) / w_norm  # Distance signée
                projections.append(proj_dist)

            projections = np.array(projections)

            # Trier par projection pour un meilleur visuel
            sort_idx = np.argsort(projections)
            projections_sorted = projections[sort_idx]
            stabilities_sorted = stabilities[sort_idx]
            y_data_sorted = y_data.flatten()[sort_idx]

            # Points selon leur projection sur w
            for i, (proj, stab, y_true) in enumerate(zip(projections_sorted,
                                                         stabilities_sorted,
                                                         y_data_sorted)):
                color = 'blue' if y_true == 1 else 'red'
                marker = 'x' if stab < 0 else 'o'  # croix si mal classé
                ax2.scatter(proj, i, c=color, marker=marker, s=50, alpha=0.7)

            # Ligne à projection=0
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)

            ax2.set_xlabel('Projection sur w (distance signée)')
            ax2.set_ylabel('Index des exemples (triés)')
            ax2.set_title('Projection des points sur le vecteur w')
            ax2.grid(True, alpha=0.3)

            # Ajouter histogramme des projections
            ax2_hist = ax2.twinx()
            ax2_hist.hist(projections, bins=30, alpha=0.3, color='gray',
                          orientation='horizontal')
            ax2_hist.set_ylabel('Fréquence')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def _plot_pca_geometric(self, X_data, y_data, stabilities, title):
        """Utilise PCA pour projeter en 2D quand N>2."""
        # Réduction à 2 dimensions avec PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_data)

        fig = plt.figure(figsize=(12, 5))

        # Graphique 1: Points dans l'espace PCA
        ax1 = fig.add_subplot(121)
        scatter = ax1.scatter(X_2d[:, 0], X_2d[:, 1],
                              c=stabilities, cmap='coolwarm',
                              s=100, alpha=0.8, edgecolors='black')

        ax1.set_xlabel('PCA 1 ({:.1f}% variance)'.format(pca.explained_variance_ratio_[0] * 100))
        ax1.set_ylabel('PCA 2 ({:.1f}% variance)'.format(pca.explained_variance_ratio_[1] * 100))
        ax1.set_title('Projection PCA des données')
        plt.colorbar(scatter, ax=ax1, label='Stabilité γ')
        ax1.grid(True, alpha=0.3)

        # Graphique 2: Distance à la frontière vs stabilité
        ax2 = fig.add_subplot(122)

        # Calculer la "distance" approximative (produit scalaire)
        X_with_bias = np.c_[np.ones(X_data.shape[0]), X_data]
        distances = np.dot(X_with_bias, self.w) / np.linalg.norm(self.w[1:])

        ax2.scatter(distances, stabilities, c=y_data.flatten(),
                    cmap='viridis', s=80, alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2,
                    label='γ=0 (frontière)')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1,
                    label='w·x=0')

        ax2.set_xlabel('Distance signée à la frontière (w·x / ||w||)')
        ax2.set_ylabel('Stabilité γ')
        ax2.set_title('Stabilité vs distance à la frontière')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'{title} - PCA projection (N={X_data.shape[1]})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_validation_errors_history(self, title="Historique des erreurs"):
        """Trace l'historique des erreurs de validation par époque."""
        if not self.error_validation_history:
            print("Aucune donnée de validation disponible.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.error_validation_history, label='Erreurs par époque', color='blue', marker='o')
        plt.xlabel('Époque')
        plt.ylabel('Nombre d\'erreurs')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def afficher_graphique(self, X, y, name):
        """
        Affiche les données et la frontière de décision pour 2D.

        Parameters:
        -----------
        X : array (n_samples, n_features)
            Données (doit avoir 2 features)
        y : array (n_samples,)
            Labels
        name : str
            Nom pour le titre du graphique
        """
        # Vérifications
        if X.shape[1] > 2:
            print("Affichage disponible uniquement pour 2 caractéristiques (x1, x2).")
            return
        if self.w is None or len(self.w) != 3:
            print("Poids non initialisés ou de mauvaise dimension (attendu: 3 avec biais).")
            return

        # Aplatissement de y si nécessaire
        y_flat = y.reshape(-1)

        plt.figure(figsize=(6, 5))

        # Tracé des points par classe
        class0 = y_flat == 0
        class1 = y_flat == 1
        plt.scatter(X[class0, 0], X[class0, 1], color='red', marker='o', label='Classe 0')
        plt.scatter(X[class1, 0], X[class1, 1], color='blue', marker='x', label='Classe 1')

        # Droite de décision: w0 + w1*x1 + w2*x2 = 0 -> x2 = -(w0 + w1*x1)/w2
        w0, w1, w2 = self.w
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        x_vals = np.linspace(x_min - 0.5, x_max + 0.5, 200)

        plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

        if np.isclose(w2, 0.0):
            # Cas frontière verticale: w0 + w1*x1 = 0 -> x1 = -w0 / w1
            if not np.isclose(w1, 0.0):
                x_vert = -w0 / w1
                y_min, y_max = X[:, 1].min(), X[:, 1].max()
                plt.plot([x_vert, x_vert], [y_min, y_max], 'k--', label='Frontière de décision')
            else:
                print("Frontière non définie (w1 et w2 proches de 0).")
        else:
            y_vals = -(w0 + w1 * x_vals) / w2
            plt.plot(x_vals, y_vals, 'k--', label='Frontière de décision')

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'frontière de décision de {name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_learning_history(self):
        """
        Retourne l'historique complet de l'apprentissage.

        Returns:
        --------
        dict : Historique contenant erreurs, meilleures erreurs, etc.
        """
        return {
            'errors_history': self.errors_history,
            'best_errors_history': self.best_errors_history,
            'validation_errors_history': self.error_validation_history,
            'epochs': self.epoch,
            'final_weights': self.w.copy() if self.w is not None else None
        }


class GenerateLS:
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.perceptron_maitre = None

    @staticmethod
    def sign(h):
        return 1 if h >= 0 else 0

    @staticmethod
    def add_biais(n, P):
        X = np.random.random((P, n))
        X = np.c_[np.ones(P), X]
        return X

    @staticmethod
    def remove_biais(X):
        return X[:, 1:]

    def get_ls(self):
        # Génération du perceptron maître
        self.perceptron_maitre = np.random.randn(self.n + 1)
        x_ls = GenerateLS.add_biais(self.n, self.p)  # array P x (n+1)
        y = np.zeros((self.p, 1))

        for i in range(self.p):
            h = np.dot(x_ls[i], self.perceptron_maitre)
            y[i] = GenerateLS.sign(h)

        return GenerateLS.remove_biais(x_ls), y
