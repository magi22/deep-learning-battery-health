# Prédiction du SoH des batteries avec un LSTM

Petit projet de prédiction du **SoH (State of Health)** des batteries à partir de mesures comme la tension, le courant, la température, le SoC et le numéro de cycle.

L’idée ici n’est pas juste d’entraîner un modèle, mais aussi de **mieux comprendre la base**, de **préparer les données proprement**, puis de **voir ce que le LSTM arrive vraiment à apprendre**.

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

Le projet peut être déployé facilement sur **Streamlit Community Cloud**.

Il faut juste :

1. mettre les fichiers sur GitHub ;
2. connecter le repo à Streamlit Cloud ;
3. choisir `app.py` comme fichier principal ;
4. lancer le déploiement.


