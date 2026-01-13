import numpy as np
from sklearn.preprocessing import StandardScaler


class Minimerror:
    """
    Implémentation de Minimerror avec sélection intelligente des exemples.
    Compatible avec tout dataset linéairement séparable.
    """

    def __init__(self, T=1.0, learning_rate=0.05, init_method="hebb",
                 hebb_noise=0.0, normalize_weights=True, scale_inputs=True,
                 momentum=0.0, min_lr_ratio=0.001):

        self.T = T
        self.learning_rate = learning_rate  # lr peut varier
        self.original_lr = learning_rate
        self.min_lr_ratio = min_lr_ratio

        self.momentum = momentum  #
        self.init_method = init_method
        self.hebb_noise = hebb_noise
        self.normalize_weights = normalize_weights
        self.scale_inputs = scale_inputs

        self.w = None
        self.scaler = StandardScaler() if scale_inputs else None
        self.velocity = None

        self.history = {
            'weights': [], 'error': [], 'cost': [],
            'T': [], 'stabilities': [], 'n_used_samples': []
        }

    # --------------------------------------------------
    # Méthodes utilitaires
    # --------------------------------------------------

    def _add_bias(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))])  # le biais est ajouté à la dernière position

    def _prepare_labels(self, y):
        """Convertit les labels en {-1, 1}."""
        y = np.asarray(y)
        unique = np.unique(y)

        if len(unique) == 2:
            # Si déjà binaire, assurer que c'est {-1, 1}
            if set(unique) == {-1, 1}:
                return y
            elif set(unique) == {0, 1}:
                return 2 * y - 1
            else:
                # Mapper les deux valeurs extrêmes à {-1, 1}
                return np.where(y == unique[0], -1, 1)
        else:
            raise ValueError("Les labels doivent avoir exactement 2 classes")

    # --------------------------------------------------
    # Initialisation
    # --------------------------------------------------

    def initialize_weights(self, X, y):
        Xb = self._add_bias(X)

        if self.init_method == "hebb":
            self.w = np.sum(Xb * y[:, None], axis=0)
            if self.hebb_noise > 0:
                self.w += self.hebb_noise * np.random.randn(*self.w.shape)
        elif self.init_method == "random":
            self.w = 0.01 * np.random.randn(Xb.shape[1])
        else:
            raise ValueError("init_method doit être 'hebb' ou 'random'")

        if self.normalize_weights:
            self.w /= np.linalg.norm(self.w) + 1e-12  # Purement numérique pour éviter une division par zéro

        # Initialiser velocity pour momentum
        if self.momentum > 0:
            self.velocity = np.zeros_like(self.w)

    # --------------------------------------------------
    # Calculs de base
    # --------------------------------------------------

    def compute_stability(self, X, y):
        """Calcule la stabilité sur des données DÉJÀ PRÉPARÉES (scalées)."""
        # PAS de scaler.transform ici !
        Xb = self._add_bias(X)
        norm = np.linalg.norm(self.w) + 1e-12
        return y * (Xb @ self.w) / norm

    def compute_test_stabilities(self, X_test, y_test):
        """Calcule les stabilités pour le test."""
        # Il faut scaler les données de test avant le calcul mathématique
        if self.scale_inputs:
            X_test = self.scaler.transform(X_test)  # Transform uniquement

        return self.compute_stability(X_test, y_test)

    def compute_cost(self, X, y):
        gamma = self.compute_stability(X, y)
        return np.sum(0.5 * (1 - np.tanh(gamma / (2 * self.T))))

    # --------------------------------------------------------
    # Sélection intelligente des exemples pour le gradient
    # -------------------------------------------------------

    def _get_relevant_samples(self, X, y, method='auto', threshold=None):
        """
        X doit être déjà scalé (si scale_inputs=True).
        """
        if method == 'all':
            return np.ones(len(y), dtype=bool)

        # compute_stability doit être safe (voir point A)
        gamma = self.compute_stability(X, y)

        if method == 'errors':
            # UTILISER _predict_internal AU LIEU DE predict
            y_pred = self._predict_internal(X)
            return y_pred != y

        elif method == 'boundary':
            if threshold is None:
                threshold = max(0.5, self.T)
            return np.abs(gamma) < threshold

        elif method == 'auto':
            # UTILISER _predict_internal ICI AUSSI
            y_pred = self._predict_internal(X)
            error_mask = y_pred != y

            n_errors = np.sum(error_mask)
            if n_errors > 0 and n_errors < len(y) * 0.3:
                return error_mask

            boundary_threshold = max(1.0, self.T * 2)
            boundary_mask = np.abs(gamma) < boundary_threshold

            if np.sum(boundary_mask) > len(y) * 0.1:
                return boundary_mask

            return np.ones(len(y), dtype=bool)

    # --------------------------------------------------
    # Gradient amélioré avec sélection
    # --------------------------------------------------

    def _compute_selective_gradient(self, X, y, method='auto', threshold=None):
        """Gradient calculé seulement sur les exemples pertinents."""
        Xb = self._add_bias(X)

        # Identifier les exemples pertinents
        mask = self._get_relevant_samples(X, y, method, threshold)

        if np.sum(mask) == 0:
            return np.zeros_like(self.w)

        # Gradient seulement sur le sous-ensemble
        Xb_selected = Xb[mask]
        y_selected = y[mask]

        n_used = np.sum(mask)
        self.history['n_used_samples'].append(n_used)

        if len(y_selected) == 0:
            return np.zeros_like(self.w)

        # Calcul standard sur le sous-ensemble
        gamma = y_selected * (Xb_selected @ self.w)
        if self.normalize_weights:
            norm = np.linalg.norm(self.w) + 1e-12
            gamma = gamma / norm

        z = gamma / (2 * self.T)
        t = np.tanh(z)
        factor = 1 - t ** 2

        grad = -(y_selected[:, None] * Xb_selected) * factor[:, None]
        grad = np.sum(grad, axis=0) / (4 * self.T)

        # Ajuster l'échelle si on utilise peu d'exemples
        if n_used < len(y) * 0.1:
            grad = grad * (len(y) / n_used)  # Ré-échelle

        return grad

    def compute_gradient(self, X, y, method='auto', threshold=None):
        """
        Gradient avec sélection optionnelle des exemples.

        Parameters:
        -----------
        method : 'auto', 'all', 'errors', 'boundary'
            Méthode de sélection des exemples
        threshold : float
            Seuil pour la méthode 'boundary'
        """
        if method == 'all':
            # Ancienne méthode (tous les exemples)
            Xb = self._add_bias(X)
            gamma = self.compute_stability(X, y)
            z = gamma / (2 * self.T)
            t = np.tanh(z)
            factor = 1 - t ** 2
            grad = -(y[:, None] * Xb) * factor[:, None]
            return np.sum(grad, axis=0) / (4 * self.T)

        else:
            # Nouvelle méthode avec sélection
            return self._compute_selective_gradient(X, y, method, threshold)

    # --------------------------------------------------
    # Entraînement
    # --------------------------------------------------

    def train(self, X, y, epochs=200, anneal=True, T_final=0.01,
              gradient_method='auto', early_stopping=True, verbose=True):
        """
        Entraînement avec options avancées.

        Parameters:
        -----------
        gradient_method : str
            Comment sélectionner les exemples pour le gradient
        early_stopping : bool
            Arrêter quand Ea = 0
        verbose : bool
            Afficher les informations
        """
        y = self._prepare_labels(y)

        if self.scale_inputs:
            X = self.scaler.fit_transform(X)

        self.initialize_weights(X, y)

        T0 = self.T
        best_error = float('inf')
        best_weights = self.w.copy()
        min_lr = self.learning_rate * self.min_lr_ratio

        for epoch in range(epochs):
            if anneal:  # Recuit déterministe (diminution de T)
                self.T = T0 - (T0 - T_final) * (epoch / epochs)

            # Gradient avec la méthode choisie
            grad = self.compute_gradient(X, y, method=gradient_method)

            # Mise à jour avec momentum
            if self.momentum > 0:
                self.velocity = (self.momentum * self.velocity
                                 - self.learning_rate * grad)
                self.w += self.velocity
            else:
                self.w -= self.learning_rate * grad

            # Normalisation
            if self.normalize_weights:
                norm = np.linalg.norm(self.w)
                if norm > 1e-12:  # éviter la division par zéro
                    self.w /= norm

            # Décay très lent du LR
            if epoch > epochs // 3:
                self.learning_rate = max(self.learning_rate * 0.9999, min_lr)

            # Évaluation
            y_pred = self._predict_internal(X)
            error_count = np.sum(y_pred != y)
            error_rate = error_count / len(y)
            cost = self.compute_cost(X, y)

            # Sauvegarde des meilleurs poids
            if error_count < best_error:
                best_error = error_count
                best_weights = self.w.copy()

            # Historique
            self.history["error"].append(error_rate)
            self.history["cost"].append(cost)
            self.history["T"].append(self.T)
            self.history["weights"].append(self.w.copy())
            self.history["stabilities"].append(self.compute_stability(X, y))

            # Affichage
            if verbose and (1 or epoch == epochs - 1 or error_count == 0):
                n_used = self.history['n_used_samples'][-1] if self.history['n_used_samples'] else len(y)
                print(f"Epoch {epoch:4d} | "
                      f"Erreur = {error_count:3d}/{len(y):3d} | "
                      f"Coût = {cost:7.3f} | "
                      f"T = {self.T:.3f} | "
                      f"Utilisé {n_used:3d}/{len(y):3d}")

            # Early stopping si Ea = 0
            if early_stopping and error_count == 0:
                if verbose:
                    print(f"Convergence à Ea = 0 (époque {epoch})")
                break

        # Restaurer les meilleurs poids et réinitialiser LR
        self.w = best_weights
        self.learning_rate = self.original_lr

        return self.history

    # --------------------------------------------------
    # Prédiction et évaluation
    # --------------------------------------------------

    def _predict_internal(self, X_already_scaled):
        """Méthode helper pour prédire sans toucher au scaler (utilisée dans train)."""
        Xb = self._add_bias(X_already_scaled)
        scores = Xb @ self.w
        y_pred = np.sign(scores)
        y_pred[y_pred == 0] = 1
        return y_pred

    def predict(self, X):
        """Prédiction pour l'utilisateur final (Test)."""
        if self.scale_inputs:
            # ICI : TRANSFORM UNIQUEMENT (pas de fit !)
            # On utilise les stats apprises pendant le train
            X = self.scaler.transform(X)

        return self._predict_internal(X)

    def force_perfect_separation(self, X, y, max_iterations=20000):
        """
        Force Ea = 0 en ajustant spécifiquement les points mal classés.
        """
        print("\n" + "=" * 60)
        print("FORÇAGE DE Ea = 0")
        print("=" * 60)

        # 1. GESTION DU SCALING (Comme dans train/predict)
        if self.scale_inputs:
            # On transforme X une fois pour toutes au début
            # On suppose que le scaler a déjà été fit pendant le train()
            X_curr = self.scaler.transform(X)
        else:
            X_curr = X

        # On s'assure que y est bien formaté
        y = self._prepare_labels(y)

        for iteration in range(max_iterations):
            # 2. Utilisation de la méthode INTERNE (sur données scalées)
            y_pred = self._predict_internal(X_curr)

            error_indices = np.where(y_pred != y)[0]
            n_errors = len(error_indices)

            if n_errors == 0:
                print(f"Ea = 0 atteint à l'itération {iteration} !")
                return True

            if iteration % 100 == 0:
                print(f"Itération {iteration}: {n_errors} erreurs")

            # Pour chaque erreur, ajuster légèrement les poids
            for idx in error_indices:
                # X_curr[idx] est maintenant correctement scalé
                x_vec = np.hstack([X_curr[idx], 1.0])  # Avec biais
                true_class = y[idx]

                # Calculer combien il manque
                current_score = x_vec @ self.w
                needed_change = abs(current_score) + 0.01  # Marge de sécurité

                # Ajustement proportionnel
                adjustment = 0.001 * true_class * x_vec * needed_change

                # Appliquer
                self.w += adjustment

            # Renormaliser
            if self.normalize_weights:
                norm = np.linalg.norm(self.w)
                if norm > 1e-12:
                    self.w /= norm

            # Vérifier si corrigé
            if iteration % 10 == 0:
                # Vérification interne rapide
                new_errors = np.sum(self._predict_internal(X_curr) != y)
                print(f"  Après ajustement: {new_errors} erreurs")

        print(f"Impossible de forcer Ea = 0 après {max_iterations} itérations")
        return False


class MinimerrorTwoTemp:
    """
    Minimerror avec deux températures distinctes: beta_plus et beta_minus.
    """

    def __init__(self, beta0=1.0, rapport_temperature=10, learning_rate=0.05,
                 init_method="hebb", hebb_noise=0.0, normalize_weights=True,
                 scale_inputs=True, momentum=0.0, min_lr_ratio=0.001):

        self.beta_plus = beta0
        self.rapport_temperature = rapport_temperature
        self.beta_minus = self.beta_plus / rapport_temperature
        self.learning_rate = learning_rate
        self.init_method = init_method
        self.hebb_noise = hebb_noise
        self.normalize_weights = normalize_weights
        self.scale_inputs = scale_inputs
        self.momentum = momentum
        self.min_lr_ratio = min_lr_ratio

        self.w = None
        self.scaler = StandardScaler() if scale_inputs else None
        self.velocity = None
        self.original_lr = learning_rate

        self.history = {
            'weights': [], 'error': [], 'cost': [],
            'beta_plus': [], 'beta_minus': [], 'stabilities': []
        }

    # --------------------------------------------------
    # Méthodes utilitaires
    # --------------------------------------------------

    def _add_bias(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def _prepare_labels(self, y):
        """Convertit les labels en {-1, 1} de manière robuste."""
        y = np.asarray(y)
        unique = np.unique(y)

        if len(unique) == 2:
            if set(unique) == {-1, 1}:
                return y
            elif set(unique) == {0, 1}:
                return 2 * y - 1
            else:
                return np.where(y == unique[0], -1, 1)
        else:
            raise ValueError("Les labels doivent avoir exactement 2 classes")

    # --------------------------------------------------
    # Initialisation
    # --------------------------------------------------

    def initialize_weights(self, X, y):
        Xb = self._add_bias(X)

        if self.init_method == "hebb":
            self.w = np.sum(Xb * y[:, None], axis=0)
            if self.hebb_noise > 0:
                self.w += self.hebb_noise * np.random.randn(*self.w.shape)
        elif self.init_method == "random":
            self.w = 0.01 * np.random.randn(Xb.shape[1])
        else:
            raise ValueError("init_method doit être 'hebb' ou 'random'")

        if self.normalize_weights:
            self.w /= np.linalg.norm(self.w) + 1e-12

        if self.momentum > 0:
            self.velocity = np.zeros_like(self.w)

    # --------------------------------------------------
    # Calculs de base avec deux températures
    # --------------------------------------------------

    def compute_stability(self, X, y=None):
        """
        Calcule la stabilité (distance signée) pour chaque exemple.
        Si y est None, retourne juste les scores non signés.
        """

        Xb = self._add_bias(X)
        norm = np.linalg.norm(self.w) + 1e-12

        if y is not None:
            return y * (Xb @ self.w) / norm
        else:
            return (Xb @ self.w) / norm

    def compute_cost(self, X, y):
        """Coût avec deux températures différentes selon la classe."""
        Xb = self._add_bias(X)
        scores = Xb @ self.w

        if self.normalize_weights:
            norm = np.linalg.norm(self.w) + 1e-12
            scores = scores / norm

        gamma = y * scores

        # Appliquer la température appropriée selon la classe
        beta_values = np.where(gamma > 0, self.beta_plus, self.beta_minus)
        z = gamma * beta_values / 2
        t = np.tanh(z)

        return np.sum(0.5 * (1 - t))

    # --------------------------------------------------
    # Gradient avec deux températures
    # --------------------------------------------------
    def compute_gradient(self, X, y):
        Xb = self._add_bias(X)
        norm = np.linalg.norm(self.w) + 1e-12
        # Important : stabilité avec poids normalisés
        gamma = self.compute_stability(X, y)

        beta_values = np.where(gamma > 0, self.beta_plus, self.beta_minus)
        z = gamma * beta_values / 2
        factor = (beta_values / (4 * norm)) * (1 - np.tanh(z) ** 2)

        # Gradient "brut"
        # Somme de [ y_i * X_i - gamma_i * (w/norm) ]
        # Le terme (gamma_i * w) est la projection qui maintient la contrainte de normalisation
        term_x = y[:, None] * Xb
        term_proj = gamma[:, None] * (self.w / norm)

        grad = - np.sum(factor[:, None] * (term_x - term_proj), axis=0)
        return grad

    # --------------------------------------------------
    # Entraînement
    # --------------------------------------------------

    def train(self, X_train, y_train, epochs=200, early_stopping=True, verbose=True, beta_max=100, b=0.9):
        """Entraînement avec deux températures."""
        y_train = self._prepare_labels(y_train)

        if self.scale_inputs:
            X_train = self.scaler.fit_transform(X_train)

        self.initialize_weights(X_train, y_train)

        beta0 = self.beta_plus
        best_error = np.sum(self._predict_internal(X_train) != y_train)
        best_weights = self.w.copy()

        for epoch in range(epochs):
            self.beta_plus = beta0 + (beta_max - beta0) * (epoch / epochs)

            self.beta_minus = self.beta_plus / self.rapport_temperature

            grad = self.compute_gradient(X_train, y_train)
            self.w -= self.learning_rate * grad

            # Normalisation des poids
            if self.normalize_weights:
                norm = np.linalg.norm(self.w)
                if norm > 1e-12:
                    self.w /= norm

            # Évaluation
            y_pred = self._predict_internal(X_train)
            error_count = np.sum(y_pred != y_train)

            error_rate = error_count / len(y_train)
            cost = self.compute_cost(X_train, y_train)

            # Sauvegarde des meilleurs poids
            if error_count < best_error:
                best_error = error_count
                best_weights = self.w.copy()
            else:
                self.learning_rate *= b  # Décroissance du taux d'apprentissage

            # # Historique
            self.history["error"].append(error_rate)
            self.history["cost"].append(cost)
            self.history["beta_plus"].append(self.beta_plus)
            self.history["beta_minus"].append(self.beta_minus)
            self.history["weights"].append(self.w.copy())
            self.history["stabilities"].append(
                self.compute_stability(X_train, y_train))

            # Affichage
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1 or error_count == 0):
                print(f"Epoch {epoch:4d} | "
                      f"Erreur = {error_count:3d}/{len(y_train):3d} | "
                      f"Coût = {cost:7.3f} | "
                      f"β+ = {self.beta_plus:.2f}, β- = {self.beta_minus:.2f}")

            # Early stopping
            if early_stopping and error_count == 0:
                if verbose:
                    print(f"Convergence à Ea = 0 (époque {epoch})")
                break

        # Restaurer les meilleurs poids
        self.w = best_weights
        self.learning_rate = self.original_lr
        return self.history

    # --------------------------------------------------
    # Prédiction
    # --------------------------------------------------

    def _predict_internal(self, X_already_scaled):
        """Méthode helper pour prédire sans toucher au scaler (utilisée dans train)."""
        Xb = self._add_bias(X_already_scaled)
        scores = Xb @ self.w
        y_pred = np.sign(scores)
        y_pred[y_pred == 0] = 1
        return y_pred

    def predict(self, X):
        """Prédiction pour l'utilisateur final (Test)."""
        if self.scale_inputs:
            # ICI : TRANSFORM UNIQUEMENT (pas de fit !)
            # On utilise les stats apprises pendant le train
            X = self.scaler.transform(X)

        return self._predict_internal(X)

    # --------------------------------------------------
    # Méthodes pour la partie II
    # --------------------------------------------------

    def compute_errors(self, X_train, y_train, X_test, y_test):
        """Calcule Ea et Eg."""
        # Erreur d'apprentissage
        y_pred_train = self.predict(X_train)
        Ea = np.sum(y_pred_train != y_train)

        # Erreur de généralisation
        y_pred_test = self.predict(X_test)
        Eg = np.sum(y_pred_test != y_test)

        return Ea, Eg

    def save_weights(self, filename):
        """Sauvegarde les N+1 poids dans un fichier."""
        if self.w is None:
            raise ValueError("Le modèle n'a pas été entraîné")

        np.savetxt(filename, self.w)
        print(f"Poids sauvegardés dans {filename}")

    def compute_test_stabilities(self, X_test, y_test):
        """Calcule les stabilités pour le test."""
        # Il faut scaler les données de test avant le calcul mathématique
        if self.scale_inputs:
            X_test = self.scaler.transform(X_test)  # Transform uniquement

        return self.compute_stability(X_test, y_test)

    def get_normalized_weights(self):
        """Retourne les poids normalisés."""
        if self.w is None:
            return None
        norm = np.linalg.norm(self.w)
        return self.w / norm if norm > 0 else self.w