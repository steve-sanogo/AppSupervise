from Classs import Perceptron
import numpy as np
import pandas as pd
from tabulate import tabulate
from tools import parse_sonar_file

"""
    Variables Globales
"""
file_path = r"../data"
x_train, y_train = None, None
x_test, y_test = None, None
x_all, y_all = None, None


def question_1():
    global file_path
    global x_train, y_train
    global x_test, y_test
    global x_all, y_all

    # train, test sets for rocks
    filename = "sonar.mines"
    try:
        x_train_mines, x_test_mines, y_train_mines, y_test_mines = parse_sonar_file(filename, file_path)
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{filename}' n'a pas été trouvé.")
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}")

    # train, test sets for rocks
    filename = "sonar.rocks"
    try:
        x_train_rocks, x_test_rocks, y_train_rocks, y_test_rocks = parse_sonar_file(filename, file_path)
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{filename}' n'a pas été trouvé.")
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}")

    x_train = pd.concat([x_train_mines, x_train_rocks])
    x_test = pd.concat([x_test_mines, x_test_rocks])

    y_train = pd.concat([y_train_mines, y_train_rocks])
    y_test = pd.concat([y_test_mines, y_test_rocks])

    x_all = pd.concat([x_train, x_test])
    y_all = pd.concat([y_train, y_test])

    print(f"Dimension x_tain :({x_train.shape[0]}, {x_train.shape[1]})")

    print(f"Dimension x_test :({x_test.shape[0]}, {x_test.shape[1]})")

    print(f"Dimension x_all :({x_all.shape[0]}, {x_all.shape[1]})")

    print(y_test['class'].unique(), y_all.shape)

    # Il faut transformer les données catégorielles  M, R en 1, 0 pour les questions suivantes
    y_train['class'] = y_train['class'].map({'M': 1, 'R': 0})
    y_test['class'] = y_test['class'].map({'M': 1, 'R': 0})
    y_all['class'] = y_all['class'].map({'M': 1, 'R': 0})


""""
    2 Apprentissage sur « train ». Utiliser l’algorithme du perceptron (justifier le choix
    version batch vs online selon votre TP1) pour apprendre l’ensemble « train », puis tester
    sur l’ensemble de « test ».
    a) Calculer les erreurs d’apprentissage Ea et de généralisation Eg ;
    b) Afficher les N+1 poids W du perceptron ;
    c) Calculer les stabilités des P exemples de « test » selon la formule de gamma (distance a l’hyperplan séparateur avec les poids normés)
    d) Graphique des stabilités
"""


def question_2():

    perceptron = Perceptron(eta=0.5, max_epoch=1000)
    perceptron.train_online(x_train.to_numpy(), y_train.to_numpy())

    # Prédictions
    _ = perceptron.predict(x_test=x_test.to_numpy())

    # Erreur d'apprentissage
    # print(_)  # à supprimer
    ea = perceptron.get_ea()

    # Erreur de généralisation
    eg = perceptron.get_eg(x_test.to_numpy(), y_test.to_numpy())

    print(f"L'erreur d'apprentissage Ea : {ea}")
    print(f"L'erreur de généralisation Eg : {eg} ")

    print(f"Les n+1 poids w du perceptron : {perceptron.w}")

    print("Calcul des stabilités des P exemples de « test » selon la formule de gamma ")

    stabilities = perceptron.compute_stability(x_test.to_numpy(), y_test.to_numpy())

    print(f"Matrice de stabilité : {stabilities}")

    print(f"Stabilité moyenne : {np.mean(stabilities):.4f}")

    print("Graphique simple des stabilités")
    perceptron.plot_stability_geometric(x_test.to_numpy(), y_test.to_numpy(),
                                        title="Stabilités géométriques")

    # Pour voir juste la représentation 2D de base
    """ perceptron._plot_2d_geometric(x_train.to_numpy(), y_train.to_numpy(),
                                  perceptron.compute_stability(x_train.to_numpy(), y_train.to_numpy()),
                                  "Représentation 2D")"""

    # perceptron.plot_stability_analysis(x_train.to_numpy(), y_train, X_test, y_test)



"""
3. Apprentissage sur « test », cad inverser les ensembles : Apprendre sur l’ensemble
« test », puis généraliser sur l’ensemble « train ». Calculer a), b) et c) du point précédent2. 
"""

"""
    En réalisant l'apprentissage sur « test » on a Ea = 0 et plus de Eg (26), contre Ea = 9 et Eg = 20 la question précedente.
    On peut conclure qu'il s'agit d'un apprentissage par coeur puisque on a eu une mauvaise généralisation et un Ea = 0.
"""


def question_3():

    perceptron = Perceptron(eta=0.5, max_epoch=1000)

    perceptron.train_online(x_test.to_numpy(), y_test.to_numpy())  # Apprentissage sur « test »

    # Prédictions
    _ = perceptron.predict(x_test=x_train.to_numpy())

    # Erreur d'apprentissage
    ea = perceptron.get_ea()

    # Erreur de généralisation
    eg = perceptron.get_eg(x_train.to_numpy(), y_train.to_numpy())

    print(f"L'erreur d'apprentissage Ea : {ea}")
    print(f"L'erreur de généralisation Eg : {eg} ")

    print(f"Les n+1 poids w du perceptron : {perceptron.w}")

    print("Calcul des stabilités des P exemples de « test » selon la formule de gamma ")

    stabilities = perceptron.compute_stability(x_train.to_numpy(), y_train.to_numpy())

    print(f"Matrice de stabilité : {stabilities}")

    print(f"Stabilité moyenne : {np.mean(stabilities):.4f}")

    print("Graphique simple des stabilités")
    perceptron.plot_stability_geometric(x_train.to_numpy(), y_train.to_numpy(),
                                        title="Stabilités géométriques")


def question_4():
    # Apprentissage sur l'ensemble d'apprentissage

    perceptron_1 = Perceptron(eta=0.5, max_epoch=1000, pocket=True, pocket_threshold=10)
    perceptron_1.train_online(x_train.to_numpy(), y_train.to_numpy())

    # Prédictions
    _ = perceptron_1.predict(x_test=x_test.to_numpy())

    # Erreur d'apprentissage
    ea = perceptron_1.get_ea()

    # Erreur de généralisation
    eg = perceptron_1.get_eg(x_test.to_numpy(), y_test.to_numpy())

    print(f"L'erreur d'apprentissage Ea : {ea}")
    print(f"L'erreur de généralisation Eg : {eg} ")

    print(f"Les n+1 poids w du perceptron : {perceptron_1.w}")

    # Apprentissage sur l'ensemble de test

    perceptron_2 = Perceptron(eta=0.5, max_epoch=1000, pocket=True, pocket_threshold=10)

    perceptron_2.train_online(x_test.to_numpy(), y_test.to_numpy())  # Apprentissage sur « test »

    # Prédictions
    _ = perceptron_2.predict(x_test=x_train.to_numpy())

    # Erreur d'apprentissage
    ea = perceptron_2.get_ea()

    # Erreur de généralisation
    eg = perceptron_2.get_eg(x_train.to_numpy(), y_train.to_numpy())

    print(f"L'erreur d'apprentissage Ea : {ea}")
    print(f"L'erreur de généralisation Eg : {eg} ")

    print(f"Les n+1 poids w du perceptron : {perceptron_2.w}")


def question_4(x_train, y_train, x_test, y_test, etas=[0.01, 0.1, 0.5, 1.0],
               pocket_thresholds=[0, 5, 10], n_runs=5):
    """
    Expérimentation complète :
    1. Apprentissage sur train, test sur test
    2. Apprentissage sur test, test sur train
    3. Différentes initialisations (random vs Hebb)
    4. Différents taux d'apprentissage eta
    5. Affichage sous forme de tables
    """

    # Liste pour stocker tous les résultats
    all_results = []

    print("=" * 80)
    print("EXPÉRIMENTATION QUESTION 4")
    print("Analyse de Ea et Eg en fonction des ensembles, initialisations et η")
    print("=" * 80)

    # Configuration des expériences
    experiments = [
        {"train_set": "train", "test_set": "test", "x_train": x_train, "y_train": y_train,
         "x_test": x_test, "y_test": y_test, "desc": "Apprentissage sur TRAIN, test sur TEST"},
        {"train_set": "test", "test_set": "train", "x_train": x_test, "y_train": y_test,
         "x_test": x_train, "y_test": y_train, "desc": "Apprentissage sur TEST, test sur TRAIN"}
    ]

    # Boucle sur les configurations d'expérimentation
    for exp_idx, exp in enumerate(experiments):
        print(f"\n{'#' * 60}")
        print(f"CONFIGURATION {exp_idx + 1}: {exp['desc']}")
        print(f"{'#' * 60}")

        for eta in etas:
            for pocket_threshold in pocket_thresholds:
                for init_method in ["random", "Hebb"]:
                    # Liste pour les résultats de cette configuration (moyenne sur n_runs)
                    run_results = []

                    # Répéter l'expérience n_runs fois pour la stabilité
                    for run in range(n_runs):
                        # Création et entraînement du perceptron
                        perceptron = Perceptron(
                            eta=eta,
                            max_epoch=1000,
                            pocket=True,
                            pocket_threshold=pocket_threshold
                        )

                        # Apprentissage avec la méthode d'initialisation spécifiée
                        if init_method == "Hebb":
                            perceptron.train_online(
                                exp["x_train"].to_numpy() if hasattr(exp["x_train"], 'to_numpy') else exp["x_train"],
                                exp["y_train"].to_numpy() if hasattr(exp["y_train"], 'to_numpy') else exp["y_train"],
                                init_method="Hebb"
                            )
                        else:  # random
                            perceptron.train_online(
                                exp["x_train"].to_numpy() if hasattr(exp["x_train"], 'to_numpy') else exp["x_train"],
                                exp["y_train"].to_numpy() if hasattr(exp["y_train"], 'to_numpy') else exp["y_train"],
                                init_method="random"
                            )

                        # Calcul des erreurs
                        Ea = perceptron.get_ea()  # Erreur d'apprentissage finale

                        # Prédiction sur l'ensemble de test
                        y_pred = perceptron.predict(
                            exp["x_test"].to_numpy() if hasattr(exp["x_test"], 'to_numpy') else exp["x_test"]
                        )

                        # Erreur de généralisation
                        Eg = perceptron.get_eg(
                            exp["x_test"].to_numpy() if hasattr(exp["x_test"], 'to_numpy') else exp["x_test"],
                            exp["y_test"].to_numpy() if hasattr(exp["y_test"], 'to_numpy') else exp["y_test"]
                        )

                        # Nombre d'époques utilisées
                        epochs_used = perceptron.epoch

                        # Stockage des résultats de cette run
                        run_results.append({
                            "Ea": Ea,
                            "Eg": Eg,
                            "epochs": epochs_used,
                            "w_norm": np.linalg.norm(perceptron.w) if perceptron.w is not None else 0
                        })

                    # Calcul des moyennes sur les n_runs
                    avg_Ea = np.mean([r["Ea"] for r in run_results if r["Ea"] is not None])
                    avg_Eg = np.mean([r["Eg"] for r in run_results])
                    avg_epochs = np.mean([r["epochs"] for r in run_results])
                    avg_w_norm = np.mean([r["w_norm"] for r in run_results])

                    # Stockage des résultats agrégés
                    all_results.append({
                        "config": exp_idx + 1,
                        "description": exp["desc"],
                        "eta": eta,
                        "pocket_threshold": pocket_threshold,
                        "init_method": init_method,
                        "avg_Ea": avg_Ea,
                        "avg_Eg": avg_Eg,
                        "avg_epochs": avg_epochs,
                        "avg_w_norm": avg_w_norm,
                        "train_size": len(exp["y_train"]),
                        "test_size": len(exp["y_test"])
                    })

    # Affichage des résultats sous forme de tables
    display_results_tables(all_results)

    # Analyse des résultats
    analyze_results(all_results)

    return all_results


def display_results_tables(all_results):
    """Affiche les résultats sous différentes tables organisées"""

    print("\n" + "=" * 80)
    print("RÉSULTATS DÉTAILLÉS PAR CONFIGURATION")
    print("=" * 80)

    # 1. Tableau par configuration (train/test vs test/train)
    for config in [1, 2]:
        config_results = [r for r in all_results if r["config"] == config]

        print(f"\n{'#' * 60}")
        print(f"CONFIGURATION {config}: {config_results[0]['description']}")
        print(f"{'#' * 60}")

        # Création du tableau
        table_data = []
        headers = ["η", "Pocket thr", "Init", "Ea (avg)", "Eg (avg)", "Époques", "||w||", "Train/Test"]

        for result in config_results:
            # Formatage des données pour le tableau
            train_test_ratio = f"{result['train_size']}/{result['test_size']}"

            table_data.append([
                result["eta"],
                result["pocket_threshold"],
                result["init_method"],
                f"{result['avg_Ea']:.1f}",
                f"{result['avg_Eg']:.1f}",
                f"{result['avg_epochs']:.1f}",
                f"{result['avg_w_norm']:.2f}",
                train_test_ratio
            ])

        # Tri par η puis par méthode d'initialisation
        table_data.sort(key=lambda x: (x[0], x[2]))

        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # 2. Tableau comparatif entre Hebb et Random
    print("\n" + "=" * 80)
    print("COMPARAISON INITIALISATION HEBB vs RANDOM")
    print("=" * 80)

    # Regrouper par η et pocket_threshold
    unique_settings = set((r["eta"], r["pocket_threshold"], r["config"]) for r in all_results)

    table_data = []
    headers = ["Config", "η", "Pocket thr", "Init", "Ea", "Eg", "Diff Ea", "Diff Eg", "Gain"]

    for eta, pocket_thr, config in sorted(unique_settings):
        # Récupérer les résultats pour cette configuration
        hebb_results = [r for r in all_results if
                        r["eta"] == eta and
                        r["pocket_threshold"] == pocket_thr and
                        r["config"] == config and
                        r["init_method"] == "Hebb"]

        random_results = [r for r in all_results if
                          r["eta"] == eta and
                          r["pocket_threshold"] == pocket_thr and
                          r["config"] == config and
                          r["init_method"] == "random"]

        if hebb_results and random_results:
            hebb = hebb_results[0]
            random = random_results[0]

            # Calcul des différences
            diff_Ea = hebb["avg_Ea"] - random["avg_Ea"]
            diff_Eg = hebb["avg_Eg"] - random["avg_Eg"]

            # Détermination du "gagnant"
            if diff_Eg < 0:
                gain = "Hebb meilleur"
            elif diff_Eg > 0:
                gain = "Random meilleur"
            else:
                gain = "Égalité"

            # Formatage
            config_desc = f"C{config}" + (" (T→Te)" if config == 1 else "(Te→T)")

            table_data.append([
                config_desc,
                eta,
                pocket_thr,
                f"Hebb: {hebb['avg_Ea']:.1f}/{hebb['avg_Eg']:.1f}",
                f"Random: {random['avg_Ea']:.1f}/{random['avg_Eg']:.1f}",
                f"{hebb['avg_Ea']:.1f}",
                f"{hebb['avg_Eg']:.1f}",
                f"{diff_Ea:+.1f}",
                f"{diff_Eg:+.1f}",
                gain
            ])

    # Tableau réduit pour lisibilité
    headers_simple = ["Config", "η", "P thr", "Hebb (Ea/Eg)", "Random (Ea/Eg)", "Diff Eg", "Gain"]
    table_data_simple = [[row[0], row[1], row[2], row[3], row[4], row[8], row[9]] for row in table_data]

    print(tabulate(table_data_simple, headers=headers_simple, tablefmt="grid"))

    # 3. Tableau d'analyse par η
    print("\n" + "=" * 80)
    print("ANALYSE PAR TAUX D'APPRENTISSAGE η")
    print("=" * 80)

    analysis_by_eta = {}
    for eta in set(r["eta"] for r in all_results):
        eta_results = [r for r in all_results if r["eta"] == eta]

        # Calcul des statistiques
        avg_Ea = np.mean([r["avg_Ea"] for r in eta_results])
        avg_Eg = np.mean([r["avg_Eg"] for r in eta_results])
        std_Eg = np.std([r["avg_Eg"] for r in eta_results])
        min_Eg = np.min([r["avg_Eg"] for r in eta_results])
        max_Eg = np.max([r["avg_Eg"] for r in eta_results])

        analysis_by_eta[eta] = {
            "avg_Ea": avg_Ea,
            "avg_Eg": avg_Eg,
            "std_Eg": std_Eg,
            "min_Eg": min_Eg,
            "max_Eg": max_Eg
        }

    # Affichage du tableau
    table_eta = []
    headers_eta = ["η", "Ea moyen", "Eg moyen", "Std Eg", "Min Eg", "Max Eg", "Recommandation"]

    for eta, stats in sorted(analysis_by_eta.items()):
        # Recommandation basée sur Eg
        if stats["avg_Eg"] <= min(stats["avg_Eg"] for stats in analysis_by_eta.values()):
            recommendation = "⭐ OPTIMAL"
        elif stats["std_Eg"] <= min(s["std_Eg"] for s in analysis_by_eta.values()):
            recommendation = "Stable"
        else:
            recommendation = "-"

        table_eta.append([
            eta,
            f"{stats['avg_Ea']:.1f}",
            f"{stats['avg_Eg']:.1f}",
            f"{stats['std_Eg']:.2f}",
            f"{stats['min_Eg']:.1f}",
            f"{stats['max_Eg']:.1f}",
            recommendation
        ])

    print(tabulate(table_eta, headers=headers_eta, tablefmt="grid"))


def analyze_results(all_results):
    """Analyse et interprète les résultats"""

    print("\n" + "=" * 80)
    print("ANALYSE ET INTERPRÉTATION DES RÉSULTATS")
    print("=" * 80)

    # 1. Impact de l'échange train/test
    config1_results = [r for r in all_results if r["config"] == 1]
    config2_results = [r for r in all_results if r["config"] == 2]

    avg_Eg_config1 = np.mean([r["avg_Eg"] for r in config1_results])
    avg_Eg_config2 = np.mean([r["avg_Eg"] for r in config2_results])

    print(f"\n1. IMPACT DE L'ÉCHANGE TRAIN/TEST:")
    print(f"   • Eg moyen (train→test): {avg_Eg_config1:.2f}")
    print(f"   • Eg moyen (test→train): {avg_Eg_config2:.2f}")
    print(f"   • Différence: {abs(avg_Eg_config1 - avg_Eg_config2):.2f}")

    if avg_Eg_config1 < avg_Eg_config2:
        print("   → Meilleure généralisation quand on apprend sur TRAIN et teste sur TEST")
    else:
        print("   → Meilleure généralisation quand on apprend sur TEST et teste sur TRAIN")

    # 2. Impact de l'initialisation
    hebb_results = [r for r in all_results if r["init_method"] == "Hebb"]
    random_results = [r for r in all_results if r["init_method"] == "random"]

    avg_Eg_hebb = np.mean([r["avg_Eg"] for r in hebb_results])
    avg_Eg_random = np.mean([r["avg_Eg"] for r in random_results])

    print(f"\n2. IMPACT DE L'INITIALISATION:")
    print(f"   • Eg moyen (Hebb): {avg_Eg_hebb:.2f}")
    print(f"   • Eg moyen (Random): {avg_Eg_random:.2f}")
    print(f"   • Gain relatif: {(avg_Eg_random - avg_Eg_hebb) / avg_Eg_random * 100:.1f}%")

    if avg_Eg_hebb < avg_Eg_random:
        print("   → Initialisation Hebb améliore la généralisation")
    else:
        print("   → Initialisation Random est meilleure")

    # 3. Impact de η
    print(f"\n3. IMPACT DU TAUX D'APPRENTISSAGE η:")
    for eta in sorted(set(r["eta"] for r in all_results)):
        eta_results = [r for r in all_results if r["eta"] == eta]
        avg_Eg = np.mean([r["avg_Eg"] for r in eta_results])
        std_Eg = np.std([r["avg_Eg"] for r in eta_results])
        print(f"   • η={eta}: Eg={avg_Eg:.2f} ± {std_Eg:.2f}")

    # 4. Impact du seuil Pocket
    print(f"\n4. IMPACT DU SEUIL POCKET:")
    for thr in sorted(set(r["pocket_threshold"] for r in all_results)):
        thr_results = [r for r in all_results if r["pocket_threshold"] == thr]
        avg_Ea = np.mean([r["avg_Ea"] for r in thr_results])
        avg_Eg = np.mean([r["avg_Eg"] for r in thr_results])
        avg_epochs = np.mean([r["avg_epochs"] for r in thr_results])
        print(f"   • Seuil={thr}: Ea={avg_Ea:.1f}, Eg={avg_Eg:.2f}, Époques={avg_epochs:.1f}")

    # 5. Recommandations finales
    print(f"\n5. RECOMMANDATIONS:")

    # Trouver la meilleure configuration globale
    best_result = min(all_results, key=lambda x: x["avg_Eg"])

    print(f"   • Meilleure configuration:")
    print(f"     - Ensemble: {best_result['description']}")
    print(f"     - η: {best_result['eta']}")
    print(f"     - Initialisation: {best_result['init_method']}")
    print(f"     - Seuil Pocket: {best_result['pocket_threshold']}")
    print(f"     - Résultat: Ea={best_result['avg_Ea']:.1f}, Eg={best_result['avg_Eg']:.2f}")


# Fonction pour exécuter l'expérimentation avec vos données
def run_experiment_with_your_data():
    """
    Exemple d'utilisation avec vos données spécifiques
    """
    # Supposons que vous avez déjà chargé vos données
    # x_train, y_train, x_test, y_test

    # Exemple avec des données synthétiques si vous n'avez pas de vraies données
    if 'x_train' not in locals():
        print("Génération de données synthétiques pour démonstration...")
        n_samples = 200
        n_features = 10

        # Génération de données aléatoires
        x_train = np.random.randn(n_samples // 2, n_features)
        y_train = np.random.randint(0, 2, n_samples // 2)

        x_test = np.random.randn(n_samples // 2, n_features)
        y_test = np.random.randint(0, 2, n_samples // 2)

    # Exécution de l'expérimentation
    results = question_4(x_train, y_train, x_test, y_test)

    # Visualisation supplémentaire
    plot_results_summary(results)

    return results


def plot_results_summary(results):
    """Visualisation graphique des résultats (optionnel)"""
    try:
        import matplotlib.pyplot as plt

        # Préparation des données
        etas = sorted(set(r["eta"] for r in results))

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Eg vs η par méthode d'initialisation
        ax = axes[0, 0]
        for init_method in ["Hebb", "random"]:
            init_results = [r for r in results if r["init_method"] == init_method]
            eta_vals = []
            eg_vals = []
            for eta in etas:
                eta_results = [r for r in init_results if r["eta"] == eta]
                if eta_results:
                    eta_vals.append(eta)
                    eg_vals.append(np.mean([r["avg_Eg"] for r in eta_results]))
            ax.plot(eta_vals, eg_vals, 'o-', label=f"Init {init_method}")

        ax.set_xlabel("η")
        ax.set_ylabel("Eg moyen")
        ax.set_title("Erreur de généralisation vs η")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Comparaison Ea vs Eg
        ax = axes[0, 1]
        config1_results = [r for r in results if r["config"] == 1]
        config2_results = [r for r in results if r["config"] == 2]

        ax.scatter([r["avg_Ea"] for r in config1_results],
                   [r["avg_Eg"] for r in config1_results],
                   alpha=0.5, label="Train→Test")
        ax.scatter([r["avg_Ea"] for r in config2_results],
                   [r["avg_Eg"] for r in config2_results],
                   alpha=0.5, label="Test→Train")

        ax.set_xlabel("Ea")
        ax.set_ylabel("Eg")
        ax.set_title("Relation Ea vs Eg")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Impact du seuil Pocket
        ax = axes[1, 0]
        pocket_thresholds = sorted(set(r["pocket_threshold"] for r in results))
        for eta in etas[:2]:  # Juste 2 valeurs de η pour la lisibilité
            eta_results = [r for r in results if r["eta"] == eta]
            thr_vals = []
            eg_vals = []
            for thr in pocket_thresholds:
                thr_results = [r for r in eta_results if r["pocket_threshold"] == thr]
                if thr_results:
                    thr_vals.append(thr)
                    eg_vals.append(np.mean([r["avg_Eg"] for r in thr_results]))
            ax.plot(thr_vals, eg_vals, 'o-', label=f"η={eta}")

        ax.set_xlabel("Seuil Pocket")
        ax.set_ylabel("Eg moyen")
        ax.set_title("Impact du seuil Pocket sur Eg")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Norme des poids
        ax = axes[1, 1]
        norm_data = {}
        for init_method in ["Hebb", "random"]:
            init_results = [r for r in results if r["init_method"] == init_method]
            norm_data[init_method] = [r["avg_w_norm"] for r in init_results]

        ax.boxplot([norm_data["Hebb"], norm_data["random"]],
                   labels=["Hebb", "Random"])
        ax.set_ylabel("||w|| moyen")
        ax.set_title("Distribution de la norme des poids")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib non disponible pour les visualisations")


def question_5():
    global x_all, y_all

    # Numpy arrays
    Xn = x_all.to_numpy()
    yn = y_all.to_numpy()

    # Entraînement perceptron sur l'ensemble L
    perceptron = Perceptron(eta=0.5, max_epoch=1000)
    perceptron.train_online(Xn, yn)
    Ea = perceptron.get_ea()

    # Critère LS: Ea == 0 signifie linéairement séparables (pour ce perceptron)
    is_ls = (Ea == 0)

    print("\n=== Question 5: Apprentissage sur L = Train + Test ===")
    print(f"Taille L: X={x_all.shape}, y={len(yn)}")
    print(f"Ea (sur L): {Ea}")
    print(f"L est-il LS ? {'Oui (Ea=0)' if is_ls else 'Non (Ea>0)'}")


def question_6(n_runs=10, eta=0.5, max_epoch=1000):
    global x_all, y_all

    Xn = x_all.to_numpy()
    yn = y_all.to_numpy()

    # Stats accumulées
    Ea_list, Ev_list, Et_list = [], [], []

    for run in range(n_runs):
        # Split aléatoire LA(50%), LV(20%), LT(30%)
        n = len(yn)
        idx = np.arange(n)
        np.random.shuffle(idx)

        nA = int(0.5 * n)
        nV = int(0.2 * n)

        idxA = idx[:nA]
        idxV = idx[nA:nA + nV]
        idxT = idx[nA + nV:]

        XA, yA = Xn[idxA], yn[idxA]
        XV, yV = Xn[idxV], yn[idxV]
        XT, yT = Xn[idxT], yn[idxT]

        # Early Stopping: on garde les meilleurs poids sur LV
        p = Perceptron(eta=eta, max_epoch=max_epoch, early_stopping=True)
        p.train_online(XA, yA, x_val=XV, y_val=yV)

        p.plot_validation_errors_history()
        # Erreurs finales
        _ = p.predict(x_test=XA)
        Ea = p.get_eg(XA, yA)

        _ = p.predict(x_test=XV)
        Ev = p.get_eg(XV, yV)

        _ = p.predict(x_test=XT)
        Et = p.get_eg(XT, yT)

        Ea_list.append(Ea)
        Ev_list.append(Ev)
        Et_list.append(Et)

        print(f"Run {run + 1}/{n_runs} -> Ea={Ea}, Ev={Ev}, Et={Et}")

    # Statistiques
    print("\n=== Early Stopping sur L (Train+Test) ===")
    print(f"n_runs={n_runs}, eta={eta}, max_epoch={max_epoch}")
    print(f"Ea moyen: {np.mean(Ea_list):.2f} ± {np.std(Ea_list):.2f}")
    print(f"Ev moyen: {np.mean(Ev_list):.2f} ± {np.std(Ev_list):.2f}")
    print(f"Et moyen: {np.mean(Et_list):.2f} ± {np.std(Et_list):.2f}")


if __name__ == "__main__":
    question_1()
    question_2()
    question_3()
    question_4(x_train, y_train, x_test, y_test)
    question_5()
    question_6()






