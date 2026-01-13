import numpy as np
from MinimError import Minimerror, MinimerrorTwoTemp


class Monoplan:
    def __init__(self, N, P, H_max=10, E_max=0):
        """
        Algorithme Monoplan pour l'apprentissage incrémental (Torres-Moreno et al.)
        Construit un réseau constructif couche par couche.

        Args:
            N: Nombre d'entrées
            P: Nombre d'exemples d'apprentissage
            H_max: Nombre maximum de couches cachées (défaut: 10)
            E_max: Nombre maximum d'erreurs d'apprentissage tolérées (défaut: 0)
        """
        self.N = N
        self.P = P
        self.H_max = H_max
        self.E_max = E_max
        self.hidden_layers = []
        self.output_layer = None

    def train(self, X_train, y_train, verbose=True):
        """
        Entraîne le réseau avec l'algorithme Monoplan
        """
        h = 0  # Nombre de couches cachées
        tau = y_train.copy()
        previous_e_h = None
        stagnation_count = 0

        # Boucle principale (Construction couches cachées)
        while True:
            # Boucle interne pour ajouter une unité cachée
            while True:
                h += 1
                if verbose:
                    print(f"\n=== Construction de la couche cachée {h} ===")

                # Préparer les entrées: X_train + sorties des couches cachées précédentes
                if h == 1:
                    X_input = X_train
                else:
                    hidden_outputs = self._get_all_hidden_outputs(X_train)
                    X_input = np.column_stack([X_train, hidden_outputs])

                # NOTE : On NE rajoute PAS de biais manuellement ici (np.ones).
                # Minimerror s'en charge via _add_bias.

                # Vérifier s'il y a plus d'une classe dans tau
                unique_targets = np.unique(tau)
                if len(unique_targets) == 1:
                    if verbose:
                        print(f"Toutes les cibles sont identiques ({unique_targets[0]})")
                    hidden_unit = TrivialClassifier(unique_targets[0])
                else:
                    # Stratégie heuristique : Essayer plusieurs initialisations
                    best_unit = None
                    best_error = float('inf')

                    for attempt in range(3):
                        # Paramètres agressifs pour apprendre vite les erreurs
                        hidden_unit = Minimerror(
                            T=20.0,
                            learning_rate=0.5,
                            init_method='random' if attempt > 0 else 'hebb',
                            hebb_noise=0.1 * (attempt + 1),
                            normalize_weights=True,
                            scale_inputs=False,  # Déjà géré ou non nécessaire ici
                            momentum=0.7,
                            min_lr_ratio=0.01
                        )

                        hidden_unit.train(
                            X_input, tau,  # X_input direct
                            epochs=2000,
                            anneal=True,
                            T_final=0.1,
                            gradient_method='errors',
                            early_stopping=True,
                            verbose=False
                        )

                        # Évaluer
                        sigma_temp = np.sign(hidden_unit.predict(X_input))
                        tau_temp = tau * sigma_temp
                        e_temp = np.sum(1 - tau_temp) / 2

                        if e_temp < best_error:
                            best_error = e_temp
                            best_unit = hidden_unit

                        if e_temp < len(tau) * 0.3:  # Si assez bon, on garde
                            break

                    hidden_unit = best_unit

                # Validation de l'unité choisie
                sigma = np.sign(hidden_unit.predict(X_input))
                self.hidden_layers.append(hidden_unit)

                # Mettre les nouveaux objectifs à apprendre
                # Règle : tau_next = tau_prev * sortie_unit
                tau_next = tau * sigma

                # Vérifier si séparable (Ea=0)
                separable = np.all(tau_next == 1)

                if separable:
                    if verbose: print("Ea=0 atteint (Séparabilité linéaire trouvée)")
                    break

                # Compter erreurs restantes
                e_h = np.sum(1 - tau_next) / 2
                if verbose: print(f"Nombre d'erreurs restantes e_h = {e_h}")

                # Gestion Stagnation
                if previous_e_h is not None and e_h >= previous_e_h:
                    stagnation_count += 1
                    if verbose: print(f"Stagnation détectée ({stagnation_count})")
                else:
                    stagnation_count = 0

                # Fallback : Si stagnation, essayer modèle plus puissant (TwoTemp)
                if stagnation_count >= 3:
                    if verbose: print(">>> Stagnation : Tentative avec MinimerrorTwoTemp")
                    self.hidden_layers.pop()  # Retirer l'unité qui a échoué

                    hidden_unit = MinimerrorTwoTemp(
                        beta0=1.0,
                        rapport_temperature=5,
                        learning_rate=0.3,
                        init_method='random',
                        hebb_noise=0.2,
                        normalize_weights=True,
                        scale_inputs=False,
                        momentum=0.8
                    )

                    # On entraîne plus longtemps
                    hidden_unit.train(X_input, tau, epochs=3000, verbose=False, beta_max=50)

                    sigma = np.sign(hidden_unit.predict(X_input))
                    self.hidden_layers.append(hidden_unit)

                    tau_next = tau * sigma
                    e_h = np.sum(1 - tau_next) / 2
                    stagnation_count = 0

                previous_e_h = e_h
                if e_h == 0: break

                # Limite H_max
                if h >= self.H_max:
                    if verbose: print(f"Limite H_max atteinte ({h})")
                    break

                tau = tau_next

            # --- Apprentissage de la sortie ---
            if verbose: print(f"\n=== Apprentissage de la couche de sortie ===")

            X_hidden = self._get_all_hidden_outputs(X_train)
            # Pas de np.ones manuel ici non plus

            unique_outputs = np.unique(y_train)
            if len(unique_outputs) == 1:
                self.output_layer = TrivialClassifier(unique_outputs[0])
            else:
                self.output_layer = Minimerror(
                    T=10,
                    learning_rate=0.3,
                    init_method='hebb',
                    hebb_noise=0.1,
                    normalize_weights=True,
                    scale_inputs=False,
                    momentum=0.7,
                    min_lr_ratio=0.001
                )
                self.output_layer.train(
                    X_hidden, y_train,  # X_hidden direct
                    epochs=2000,
                    anneal=True,
                    T_final=0.01,
                    gradient_method='auto',
                    early_stopping=True,
                    verbose=False
                )

            zeta = self.output_layer.predict(X_hidden)
            tau_final = y_train * zeta
            e_zeta = np.sum(1 - tau_final) / 2

            if verbose: print(f"Erreurs finales sortie e_ζ = {e_zeta}")

            # Condition d'arrêt globale
            if h <= self.H_max and e_zeta > self.E_max:
                if verbose: print("Objectif non atteint, on continue d'ajouter des couches...\n")
                previous_e_h = None
                stagnation_count = 0
            else:
                if verbose: print(f"\n>>> Fin Monoplan (h={h}, Erreurs={e_zeta})")
                break

    def _get_all_hidden_outputs(self, X):
        """Obtenir les sorties de toutes les couches cachées concaténées"""
        if len(self.hidden_layers) == 0:
            return np.array([]).reshape(X.shape[0], 0)

        all_outputs = []
        current_input = X

        for i, layer in enumerate(self.hidden_layers):
            # Pour chaque couche, l'entrée est X + sorties précédentes
            if i > 0:
                previous_outputs = np.hstack(all_outputs)
                current_input = np.column_stack([X, previous_outputs])
            else:
                current_input = X

            # Prédiction (Minimerror ajoute le biais)
            layer_output = layer.predict(current_input)
            all_outputs.append(layer_output.reshape(-1, 1))

        return np.hstack(all_outputs)

    def predict(self, X):
        """Prédire les étiquettes pour de nouveaux exemples"""
        if not self.output_layer:
            raise ValueError("Modèle non entraîné!")

        X_hidden = self._get_all_hidden_outputs(X)
        return self.output_layer.predict(X_hidden)

    # --- NOUVELLES FONCTIONNALITÉS ---

    def get_weights(self):
        """Récupère les poids de toutes les unités (format dictionnaire)"""
        weights = {}
        for i, layer in enumerate(self.hidden_layers):
            if hasattr(layer, 'w') and layer.w is not None:
                weights[f'Hidden_{i + 1}'] = layer.w
            else:
                weights[f'Hidden_{i + 1}'] = "Trivial/Non-trainable"

        if self.output_layer and hasattr(self.output_layer, 'w'):
            weights['Output'] = self.output_layer.w
        return weights

    def compute_stability(self, X, y):
        """
        Calcule la stabilité (marge) par rapport au neurone de SORTIE.
        """
        if not self.output_layer or not hasattr(self.output_layer, 'compute_stability'):
            return None

        # On recrée l'entrée exacte vue par le neurone de sortie
        X_hidden = self._get_all_hidden_outputs(X)
        # On laisse Minimerror gérer le biais
        return self.output_layer.compute_stability(X_hidden, y)

    def save_model(self, filename):
        """Sauvegarde l'architecture et les poids"""
        with open(filename, 'w') as f:
            f.write(f"Architecture Monoplan: {len(self.hidden_layers)} couches cachées\n")
            f.write("=" * 40 + "\n")

            all_weights = self.get_weights()
            for name, w in all_weights.items():
                f.write(f"[{name}]\n")
                if isinstance(w, np.ndarray):
                    # Formatage lisible
                    w_str = np.array2string(w, precision=4, separator=', ', suppress_small=True)
                    f.write(f"{w_str}\n")
                else:
                    f.write(f"{w}\n")
                f.write("\n")
        print(f"Modèle sauvegardé dans : {filename}")


class TrivialClassifier:
    """Classificateur trivial (dummy) qui prédit toujours la même classe"""

    def __init__(self, label):
        self.label = label

    def predict(self, X):
        return np.full(X.shape[0], self.label)

    def fit(self, X, y):
        return self