import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, eta=0.01, max_epoch=500):
        self.eta = eta  # learning rate
        self.max_epoch = max_epoch
        self.w = None
        self.errors_history = []
        self.perceptron_maitre = None

    @staticmethod
    def sign(h):
        return 1 if h >= 0 else 0

    def train_with_batch(self, x_train, y_train):
        # Ajout du biais (colonne de 1)
        self.x = np.c_[np.ones(x_train.shape[0]), x_train]
        self.y = y_train.reshape(-1, 1)

        # Initialisation des poids (w0 pour biais, w1...wn pour features)
        self.w = np.random.randn(self.x.shape[1])  # modifier

        print("=========== Training in process ===========")

        for epoch in range(1, self.max_epoch + 1):
            errors = 0

            for i in range(len(self.x)):
                # Calcul de la sortie
                h = np.dot(self.x[i], self.w)
                y_pred = self.sign(h)

                if y_pred != self.y[i]:
                    # Mise à jour des poids
                    self.w += self.eta * (self.y[i] - y_pred) * self.x[i]   # .reshape(-1, 1)
                    errors += 1

            self.errors_history.append(errors)
            # print(f"Epoch {epoch}: errors = {errors}")

            if errors == 0:
                print(f"=========== Apprentissage réussi! ===========")
                print(f"Nombre d'epochs: {epoch}")
                break
        self.epoch = epoch + 1
        if errors > 0:
            print(f"=========== Arrêt après {self.max_epoch} epochs ===========")
            print(f"Dernier nombre d'erreurs: {errors}")

    def train_online(self, x_train, y_train):
        X = np.c_[np.ones(x_train.shape[0]), x_train]  # Ajout du biais
        y = y_train.reshape(-1)  # vecteur 1D

        self.w = np.random.randn(X.shape[1])  # init poids

        for epoch in range(self.max_epoch):
            errors = 0
            indices = np.random.permutation(len(X))

            for i in indices:
                h = np.dot(X[i], self.w)
                y_pred = self.sign(h)

                if y_pred != y[i]:
                    self.w += self.eta * (y[i] - y_pred) * X[i]
                    errors += 1

            self.errors_history.append(errors)

            if errors == 0:
                print(f"Convergence à l’epoch {epoch + 1}")
                break

        self.epoch = epoch + 1

        if errors > 0:
            print(f"=========== Arrêt après {self.max_epoch} epochs ===========")
            print(f"Dernier nombre d'erreurs: {errors}")

    def afficher_graphique(self, X, y, name):
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
        x_ls = GenerateLS.add_biais(self.n, self.p)   # array P x (n+1)
        y = np.zeros((self.p, 1))

        for i in range(self.p):
            h = np.dot(x_ls[i], self.perceptron_maitre)
            y[i] = GenerateLS.sign(h)

        return GenerateLS.remove_biais(x_ls), y