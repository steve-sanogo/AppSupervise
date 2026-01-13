import re
import pandas as pd
from pathlib import Path


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
