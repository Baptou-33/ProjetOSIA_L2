# Temps d'execution lors d'un from Donnees_et_RMSE import * : 16 à 21s au CREMI
# Importation des bibliothèques -----------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np



# Importation des documents de données temporelles et de test -----------------------------------------------------------------------------
history_toulouse_train = pd.read_parquet("history_toulouse_train.parquet") # Données temporelles (Index 1 : 'station', Index 2 : 'commit_at', Colonnes : 'bikes' et 'stands')
stations_toulouse = pd.read_parquet("stations_toulouse.parquet") # Données géographiques (Colonnes : 'station', 'longitude', 'latitude', 'station_id')
history_toulouse_test = pd.read_parquet("history_toulouse_test.parquet") # A compléter (Index 1 : 'station', Index 2 : 'commit_at', Colonnes : 'bikes_+15min' à 'bikes_+120min')



# Initialisation ---------------------------------------------------------------------------------------------------------------------------
# On associe chaque nom de colonne à une valeur numérique
decalage = {
    "bikes_+15min": 15,
    "bikes_+30min": 30,
    "bikes_+45min": 45,
    "bikes_+60min": 60,
    "bikes_+75min": 75,
    "bikes_+90min": 90,
    "bikes_+105min": 105,
    "bikes_+120min": 120,
}

# On s'assure que history_toulouse_test soit vide pour détecter les erreures plus facilement
history_toulouse_test = pd.DataFrame(
    np.nan,
    index=history_toulouse_test.index,
    columns=history_toulouse_test.columns
)



# RMSE ------------------------------------------------------------------------------------------------------------------------------------
# Fonction qui créé un ensemble d'apprentissage
def ensemble_apprentissage(train):
	rang_depuis_la_fin = train.groupby(level="station").cumcount(ascending=False)
	return train[rang_depuis_la_fin >= 8].copy()

# Fonction qui créé un ensemble de validation
def ensemble_validation(train):
	# On créé l'ensemble de validation avec la dernière valeur connue + les 8 à prédire
	last_9 = train.groupby(level="station").tail(9).copy()
	
	# On remet l'index sous forme de colonnes
	last_9 = last_9.reset_index()
	
	# On calule la somme cummulée
	last_9["horizon"] = last_9.groupby("station").cumcount()
	
	# On crée une ligne par station et on récupère les 8 valeurs futures
	history_toulouse_validation = (
	    last_9
	    .pivot(
	        index="station",
	        columns="horizon",
	        values="bikes"
	    )
	    .drop(columns=0)
	    .rename(columns={
	        1: "bikes_+15min",
	        2: "bikes_+30min",
	        3: "bikes_+45min",
	        4: "bikes_+60min",
	        5: "bikes_+75min",
	        6: "bikes_+90min",
	        7: "bikes_+105min",
	        8: "bikes_+120min"
	    })
	)
	
	# Récupérer la date de la dernière mesure de l'ensemble d'apprentissage par station
	cutoff_times = last_9[last_9["horizon"] == 0].set_index("station")["commit_at"]
	
	# Ajouter commit_at
	history_toulouse_validation["commit_at"] = cutoff_times
	
	# Mettre station + commit_at en MultiIndex
	history_toulouse_validation = history_toulouse_validation.reset_index().set_index(["station", "commit_at"]).sort_index()
	return history_toulouse_validation

# Fonction pour calculer le score RMSE
def rmse_calcul(rendu, validation):
	return np.sqrt(np.nanmean((rendu.values - validation.values)**2))

# Création d'un dataset de test et d'un dataset de validation pour évaluer sa performance
history_toulouse_apprentissage = ensemble_apprentissage(history_toulouse_train)
history_toulouse_validation = ensemble_validation(history_toulouse_train)