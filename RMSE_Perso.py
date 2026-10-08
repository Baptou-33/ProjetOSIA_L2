import pandas as pd
import numpy as np

history_toulouse_train = pd.read_parquet("history_toulouse_train.parquet")
history_toulouse_test = pd.read_parquet("history_toulouse_test.parquet")

# On récupère les 8 dernières valeurs
rang_depuis_la_fin = history_toulouse_train.groupby(level="station").cumcount(ascending=False)
history_toulouse_apprentissage = history_toulouse_train[rang_depuis_la_fin >= 8].copy()