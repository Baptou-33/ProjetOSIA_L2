import pandas as pd
import numpy as np

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