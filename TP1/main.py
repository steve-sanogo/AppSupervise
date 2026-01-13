from Classs import Perceptron, GenerateLS
from threads import thread_perceptron, thread_perceptron_v2
import numpy as np
from tabulate import tabulate
import time


def question_0():
    """"
        eta = 0.1
    """

    print("""
    # ***************************************
    #          Fonction ET
    # ***************************************
    """)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    print("\n===========  Version Online ===========")

    perceptron_online = Perceptron(eta=0.1)
    perceptron_online.train_online(X, y)
    perceptron_online.afficher_graphique(X, y, 'Fonction ET Online')

    print("\n===========  Version Batch ===========")
    perceptron_batch = Perceptron(eta=0.1)
    perceptron_batch.train_with_batch(X, y)

    print("""
    # ***************************************
    #          Fonction OU
    # ***************************************
    """)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])

    print("\n===========  Version Online ===========")
    perceptron_online = Perceptron(eta=0.1)
    perceptron_online.train_online(X, y)

    print("\n===========  Version Batch ===========")
    perceptron_batch = Perceptron(eta=0.1)
    perceptron_batch.train_with_batch(X, y)
    perceptron_batch.afficher_graphique(X, y, 'Fonction OU Batch')


    print("""
    # ***************************************
    #          Fonction au choix
    # ***************************************
    """)

    X = np.array([[2, 1], [0, -1], [-2, 1], [0, 2]])
    y = np.array([1, 1, 0, 0])

    print("\n===========  Version Online ===========")
    perceptron_online = Perceptron(eta=0.1)
    perceptron_online.train_online(X, y)
    perceptron_online.afficher_graphique(X,y,'Fonction au choix Online')

    print("\n===========  Version Batch ===========")
    perceptron_batch = Perceptron(eta=0.1)
    perceptron_batch.train_with_batch(X, y)


def question_1():
    """"
        eta = 0.1
    """

    ls = GenerateLS(n=2, p=20)
    x, y = ls.get_ls()

    print("x (features sans biais) :")
    print(x)
    print("\n y (classe) :")
    print(y)

    print("Vérification que l'ensemble est LS")
    perceptron_to_verify = Perceptron()
    perceptron_to_verify.train_with_batch(x, y)


# Question 2
def question_2():
    """
        number_threads : nombre de perceptron // nombre de threads à lancer
        x_n : nombre de caractéristiques
        l_p : nombre d'exemple
        eta = 0.1
        display = plot : On affiche graphiquement le perceptron maitre et ses élèves
        display = table : On affiche au format tableau les élèves du perceptron maitre et leur valeur de recouvrement
        Par defaut display == table

    """

    thread_perceptron(training_type="online", number_threads=20, eta=0.1, x_n=2, l_p=20, display="plot")

    thread_perceptron(training_type="batch", number_threads=20, eta=0.1, x_n=10, l_p=20)


# Question 3
def _ansi_bg(r, g, b):
    return f"\033[48;2;{r};{g};{b}m"

RESET = "\033[0m"


def _bg_for_R(R):
    # clamp [0,1]
    t = 0.0 if R is None else max(0.0, min(1.0, float(R)))
    # gradient bleu -> rouge
    r = int(255 * t)
    g = 0
    b = int(255 * (1.0 - t))
    return _ansi_bg(r, g, b)


def _format_cell_tabulate(IT, R):
    bg = _bg_for_R(R)
    text = "-" if (IT is None or R is None) else f"IT={IT:.2f} ; R={R:.5f}"
    # fond coloré + texte noir pour lisibilité
    return f"{bg}\033[30m{text}{RESET}"


def question_3_batch():
    eta0 = 0.1
    eta = [eta0, eta0 / 2, eta0 / 10]
    n_threads = 100
    N = [2, 10, 500, 1000, 5000]
    P = [10, 100, 200,500, 1000]
    table_results = []
    table_time = []
    print("\n===========  Version Batch ===========")
    for e in eta:
        table_e = []
        t0 = time.perf_counter()
        for N_i in N:
            for P_i in P:
                IT, R = thread_perceptron_v2(n_threads, N_i, P_i, e, "batch")
                table_e.append((N_i, P_i, IT, R))
        t1 = time.perf_counter()
        elapsed = t1 - t0
        table_results.append((e, table_e))
        table_time.append((e, elapsed))
    for eta_value, results_eta in table_results:
        print(f"\n=== Résultats pour eta = {eta_value} ===")

        # 1) Construire (N, P) -> cellule formatée avec fond coloré selon R
        matrix = {}  # { N : { P : "IT=.. ; R=.." (avec ANSI bg) } }
        for N_i, P_i, IT, R in results_eta:
            if N_i not in matrix:
                matrix[N_i] = {}
            matrix[N_i][P_i] = _format_cell_tabulate(IT, R)

        # 2) Construire le tableau pour tabulate
        header = ["N \\ P"] + [str(P_i) for P_i in P]
        table = []
        for N_i in N:
            row = [str(N_i)]
            for P_i in P:
                row.append(matrix.get(N_i, {}).get(P_i, _format_cell_tabulate(None, None)))
            table.append(row)

        # 3) Afficher avec tabulate
        print(tabulate(table, headers=header, tablefmt="fancy_grid"))

    header_time = ["eta", "Temps écoulé (s)"]
    table_time_display = [[e, f"{elapsed:.4f}"] for e, elapsed in table_time]
    print("\n=== Temps écoulé pour chaque eta ===")
    print(tabulate(
        table_time_display,
        headers=header_time,
        tablefmt="fancy_grid"
    ))


def question_3_online():

    eta0 = 0.1
    eta = [eta0, eta0 / 2, eta0 / 10]
    n_threads = 100
    N = [2, 10, 100, 500, 1000, 5000]
    P = [10, 100, 200, 500, 1000]
    table_results = []
    table_time = []
    print("\n===========  Version Online ===========")
    for e in eta:
        table_e = []
        t0 = time.perf_counter()
        for N_i in N:
            for P_i in P:
                IT, R = thread_perceptron_v2(n_threads, N_i, P_i, e, "online")
                table_e.append((N_i, P_i, IT, R))
        t1 = time.perf_counter()
        elapsed = t1 - t0
        table_results.append((e, table_e))
        table_time.append((e, elapsed))

    for eta_value, results_eta in table_results:
        print(f"\n=== Résultats pour eta = {eta_value} ===")

        # 1. Construire un dictionnaire (N, P) -> "IT=.. ; R=.."
        matrix = {}  # { N : { P : "IT=.. ; R=.." } }
        for N_i, P_i, IT, R in results_eta:
            if N_i not in matrix:
                matrix[N_i] = {}
            matrix[N_i][P_i] = _format_cell_tabulate(IT, R)

        # 2. Construire le tableau avec tabulate
        header = ["N \\ P"] + [str(P_i) for P_i in P]

        table = []
        for N_i in N:
            row = [N_i]
            for P_i in P:
                row.append(matrix[N_i].get(P_i, "-"))
            table.append(row)

        # 3. Affichage tabulaire
        print(tabulate(
            table,
            headers=header,
            tablefmt="fancy_grid"
        ))
    header_time = ["eta", "Temps écoulé (s)"]
    table_time_display = [[e, f"{elapsed:.4f}"] for e, elapsed in table_time]
    print("\n=== Temps écoulé pour chaque eta ===")
    print(tabulate(
        table_time_display,
        headers=header_time,
        tablefmt="fancy_grid"
    ))


if __name__ == "__main__":
    question_0()
    question_1()
    question_2()
    question_3_batch()
    question_3_online()