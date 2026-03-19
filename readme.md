# Prédiction du SoH des batteries avec un LSTM

Projet de prédiction du **SoH (State of Health)** des batteries à partir de mesures comme la tension, le courant, la température, le SoC et le numéro de cycle.

Il a pour but de prédire le SoH d’une batterie à partir de données séquentielles de décharge, dans une logique simple de régression avec LSTM.

## Ce que fait l’application

L’application permet de :

- charger un fichier CSV ;
- vérifier rapidement si la base est correcte ;
- visualiser quelques informations utiles ;
- créer des **fenêtres glissantes** ;
- séparer les données en **train / validation / test** ;
- entraîner un **modèle LSTM** ;
- afficher les résultats ;
- télécharger les prédictions en CSV.

## Fichiers du projet

- `app.py` : l’application Streamlit
- `requirements.txt` : les dépendances du projet
- `Projet_Batteries (1).ipynb` : le notebook de travail
- `battery_health_dataset.csv` : le fichier de données si tu veux le laisser dans le repo

## Colonnes attendues dans le CSV

Le fichier doit contenir ces colonnes :

- `Voltage_measured`
- `Current_measured`
- `Temperature_measured`
- `SoC`
- `cycle_number`
- `battery_id`
- `SoH`

## Lancer le projet en local

Installe les dépendances :

```bash
pip install -r requirements.txt
```

Puis lance l’application :

```bash
streamlit run app.py
```

## Déploiement

Sur **Streamlit Community Cloud**.

Il faut  :

1. mettre les fichiers sur GitHub ;
2. connecter le repo à Streamlit Cloud ;
3. choisir `app.py` comme fichier principal ;
4. lancer le déploiement.


