from Classs import Perceptron, GenerateLS
from tabulate import tabulate
import threading
import matplotlib.pyplot as plt
import numpy as np
import itertools


def afficher_results(results):
    """
        Affiche au format tabulaire les valeurs : "Thread", "Weights", "Epochs", "Recouvrement"

        """
    table = []
    for i, res in enumerate(results):
        table.append([
            i,
            str(res['weights']),
            res['epochs'],
            res['recouvrement']
        ])
    print(tabulate(table, headers=["Thread", "Weights", "Epochs", "Recouvrement"], tablefmt="fancy_grid"))
    print('\n')


def plot_perceptrons(p, results):
    """
    p.perceptron_maitre : vecteur numpy de poids
    results : liste de dictionnaires, chaque dict contient 'weights' (vecteur numpy)
    """

    # Vérification dimension
    d = len(p.perceptron_maitre)
    if d > 3:
        print(f"Impossible de représenter : dimension {d} > 3.")
        return

    # --- Génération du plan pour tracer les droites / plans
    x_min, x_max = -5, 5
    xx = np.linspace(x_min, x_max, 100)

    # Couleurs cycliques pour les élèves
    color_cycle = itertools.cycle(["blue", "green", "orange", "purple", "brown", "cyan"])

    plt.figure(figsize=(8, 6))

    # --- Fonction pour tracer un perceptron (hyperplan)
    def plot_line(weights, color, label):

        w = np.array(weights)

        # Cas dimension 2 : w0 + w1*x + w2*y = 0
        if len(w) == 3:
            # w0 + w1*x + w2*y = 0 → y = -(w0 + w1*x) / w2
            if w[2] == 0:
                return
            yy = -(w[0] + w[1] * xx) / w[2]
            plt.plot(xx, yy, color=color, linewidth=2, label=label)

        # Cas dimension 1 (rare) : w0 + w1*x = 0
        elif len(w) == 2:
            x0 = -w[0] / w[1]
            plt.axvline(x0, color=color, label=label)

        # Cas dimension 3 : projection sur (x1,x2)
        else:
            print("Représentation 3D -> projection sur x1-x2")
            w = w[:3]
            if w[2] == 0:
                return
            yy = -(w[0] + w[1] * xx) / w[2]
            plt.plot(xx, yy, color=color, linewidth=2, label=label)

    # --- Tracer le perceptron maître (ROUGE)
    plot_line(p.perceptron_maitre, "red", "Perceptron maître")

    # --- Tracer les perceptrons élèves
    for i, rslt in enumerate(results):
        color = next(color_cycle)
        plot_line(rslt["weights"], color, f"Élève {i + 1}")

    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("Représentation du perceptron maître et des perceptrons élèves")
    plt.grid(True)
    plt.legend(loc="upper right", fontsize=10)
    plt.show()


def worker(i, X, y, eta, results, perceptron_maitre, training_type):

    p = Perceptron(eta=eta)

    if training_type == "online":
        p.train_online(X, y)

    elif training_type == "batch":
        p.train_with_batch(X, y)

    w = p.w
    perceptron_maitre = perceptron_maitre.flatten()

    R = np.dot(perceptron_maitre, w)
    recouvrement = R / (np.linalg.norm(perceptron_maitre) * np.linalg.norm(w))

    results[i] = {
        "weights": w,
        "epochs": p.epoch,
        "recouvrement": recouvrement
    }


def thread_perceptron(training_type, number_threads, eta, x_n, l_p, display="table"):

    if training_type not in ["online", "batch"]:
        raise ValueError("Choisir soit batch soit online")

    threads = []
    results = [None] * number_threads

    p = GenerateLS(x_n, l_p)
    X, y = p.get_ls()

    t_type = training_type

    print(f"""
        # ***************************************
                 Thread with : {training_type}"
        # ***************************************
        """)

    for i in range(number_threads):
        t = threading.Thread(target=worker, args=(i, X, y, eta, results, p.perceptron_maitre, t_type))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if display == "plot":
        plot_perceptrons(p, results)

    elif display == "table":
        afficher_results(results)

    return True


# QUESTION 3
def thread_perceptron_v2(number_threads, x_n, l_p, eta, t_type):
    threads = []
    results = [None] * number_threads

    for i in range(number_threads):
        p = GenerateLS(x_n, l_p)
        X, y = p.get_ls()

        t = threading.Thread(target=worker, args=(i, X, y, eta, results, p.perceptron_maitre, t_type))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    IT_moyen, R_moyen = 0, 0

    for i in range(number_threads):
        IT_moyen += results[i]['epochs']
        R_moyen += results[i]['recouvrement']
    return IT_moyen / number_threads, R_moyen / number_threads

