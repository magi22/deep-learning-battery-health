import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as e:
    tf = None
    keras = None
    layers = None
    TF_IMPORT_ERROR = e
else:
    TF_IMPORT_ERROR = None

st.set_page_config(page_title="Prédiction du SoH des batteries", layout="wide")

REQUIRED_COLUMNS = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "SoC",
    "cycle_number",
    "battery_id",
    "SoH",
]


@st.cache_data
def load_dataset_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data
def load_dataset_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def check_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing


@st.cache_data
def prepare_cycle_summary(df: pd.DataFrame) -> pd.DataFrame:
    cycle_summary = (
        df.groupby(["battery_id", "cycle_number"])
        .agg(
            n_bins=("SoH", "size"),
            soh_nunique=("SoH", "nunique"),
            soh=("SoH", "first"),
        )
        .reset_index()
    )
    return cycle_summary


def split_batteries(df: pd.DataFrame):
    all_batteries = sorted(df["battery_id"].unique())
    if len(all_batteries) < 9:
        raise ValueError(
            "Le dataset contient trop peu de batteries pour garder la même logique de split. "
            "Il faut au moins 9 batteries."
        )
    test_batteries = all_batteries[-4:]
    val_batteries = all_batteries[-8:-4]
    train_batteries = all_batteries[:-8]
    return train_batteries, val_batteries, test_batteries


@st.cache_data
def create_windows(
    dataframe: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    battery_list: list,
    window_size: int,
):
    subset = dataframe[dataframe["battery_id"].isin(battery_list)].copy()
    subset = subset.sort_values(
        ["battery_id", "cycle_number", "SoC"],
        ascending=[True, True, False],
    )

    X_list, y_list, battery_ids = [], [], []

    for (bat_id, cycle_num), group in subset.groupby(["battery_id", "cycle_number"]):
        values = group[feature_cols].to_numpy(dtype=np.float32)
        soh = float(group[target_col].iloc[0])

        if len(values) < window_size:
            continue

        for i in range(len(values) - window_size + 1):
            X_list.append(values[i : i + window_size])
            y_list.append(soh)
            battery_ids.append(bat_id)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(battery_ids),
    )


def transform_X(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    shape = X.shape
    X_flat = X.reshape(-1, shape[-1])
    X_scaled = scaler.transform(X_flat)
    return X_scaled.reshape(shape)


def build_model(window_size: int, n_features: int, learning_rate: float):
    model = keras.Sequential(
        [
            layers.Input(shape=(window_size, n_features)),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def plot_soh_by_battery(df: pd.DataFrame):
    soh_cycle = df.groupby(["battery_id", "cycle_number"])["SoH"].first().reset_index()
    batteries_a_montrer = sorted(df["battery_id"].unique())[:4]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for bat in batteries_a_montrer:
        d = soh_cycle[soh_cycle["battery_id"] == bat]
        ax.plot(d["cycle_number"], d["SoH"], label=str(bat), linewidth=2)

    ax.axhline(80, linestyle="--", linewidth=1.5, label="Seuil 80")
    ax.set_title("Évolution du SoH sur quelques batteries")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SoH")
    ax.legend()
    fig.tight_layout()
    return fig



def plot_example_cycle(df: pd.DataFrame):
    battery_example = sorted(df["battery_id"].unique())[0]
    cycle_example = int(df[df["battery_id"] == battery_example]["cycle_number"].min())
    cycle_data = df[
        (df["battery_id"] == battery_example)
        & (df["cycle_number"] == cycle_example)
    ].copy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    cols = ["Voltage_measured", "Current_measured", "Temperature_measured", "SoC"]
    titles = ["Tension", "Courant", "Température", "SoC"]

    for ax, col, title in zip(axes.ravel(), cols, titles):
        ax.plot(cycle_data[col].values, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Bin")

    fig.suptitle(f"Exemple de cycle - {battery_example} / cycle {cycle_example}", fontsize=13)
    fig.tight_layout()
    return fig



def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["loss"], label="Train", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Validation", linewidth=2)
    axes[0].set_title("Évolution de la loss")
    axes[0].set_xlabel("Époque")
    axes[0].legend()

    axes[1].plot(history.history["mae"], label="Train", linewidth=2)
    axes[1].plot(history.history["val_mae"], label="Validation", linewidth=2)
    axes[1].set_title("Évolution de la MAE")
    axes[1].set_xlabel("Époque")
    axes[1].legend()

    fig.tight_layout()
    return fig



def plot_predictions(y_true, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(y_true, y_pred, alpha=0.35, s=12)
    lims = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    axes[0].plot(lims, lims, "--", linewidth=2)
    axes[0].set_title("SoH réel vs SoH prédit")
    axes[0].set_xlabel("SoH réel")
    axes[0].set_ylabel("SoH prédit")

    errors = y_pred - y_true
    axes[1].hist(errors, bins=40)
    axes[1].axvline(0, linestyle="--", linewidth=2)
    axes[1].set_title("Distribution des erreurs")
    axes[1].set_xlabel("Erreur")
    axes[1].set_ylabel("Fréquence")

    fig.tight_layout()
    return fig



def plot_battery_tracking(y_true, y_pred, bat_test, selected_battery):
    mask = bat_test == selected_battery
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(y_true[mask], label="SoH réel", linewidth=2)
    ax.plot(y_pred[mask], label="SoH prédit", linewidth=2, linestyle="--")
    ax.set_title(f"Suivi du SoH sur la batterie {selected_battery}")
    ax.set_xlabel("Fenêtres")
    ax.set_ylabel("SoH")
    ax.legend()
    fig.tight_layout()
    return fig


st.title("Prédiction du SoH des batteries avec un LSTM")

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Importer le fichier CSV", type=["csv"])
    window_size = st.slider("Taille de la fenêtre glissante", min_value=3, max_value=15, value=10)
    epochs = st.slider("Nombre maximum d'époques", min_value=5, max_value=60, value=30)
    batch_size = st.select_slider("Batch size", options=[16, 32, 64, 128], value=64)
    learning_rate = st.select_slider(
        "Learning rate",
        options=[0.0005, 0.001, 0.002, 0.005],
        value=0.001,
    )
    patience = st.slider("Patience (early stopping)", min_value=3, max_value=15, value=8)
    train_button = st.button("Lancer l'entraînement")

# Chargement du dataset
if uploaded_file is not None:
    df = load_dataset_from_bytes(uploaded_file.getvalue())
    dataset_label = uploaded_file.name
else:
    default_candidates = [
        Path("battery_health_dataset.csv"),
        Path("76d3946d-1198-4ef0-b7f5-3d21870da910.csv"),
    ]
    existing = [p for p in default_candidates if p.exists()]
    if not existing:
        st.info(
            "Importe ton CSV depuis la barre latérale. Si tu déploies l'app, tu peux aussi "
            "mettre le fichier CSV à la racine du projet."
        )
        st.stop()
    df = load_dataset_from_path(str(existing[0]))
    dataset_label = existing[0].name

missing = check_columns(df)
if missing:
    st.error(f"Colonnes manquantes : {missing}")
    st.stop()

if tf is None:
    st.error(
        "TensorFlow n'est pas disponible dans cet environnement. Ajoute-le dans requirements.txt.\n\n"
        f"Détail : {TF_IMPORT_ERROR}"
    )
    st.stop()

# Mise en forme de base
np.random.seed(42)
tf.random.set_seed(42)
df = df.copy()

st.success(f"Fichier chargé : {dataset_label}")

# Aperçu dataset
st.subheader("1. Vérification rapide de la base")
col1, col2, col3, col4 = st.columns(4)
cycle_summary = prepare_cycle_summary(df)
with col1:
    st.metric("Lignes", f"{df.shape[0]:,}".replace(",", " "))
with col2:
    st.metric("Colonnes", df.shape[1])
with col3:
    st.metric("Batteries", df["battery_id"].nunique())
with col4:
    st.metric("Cycles", cycle_summary.shape[0])

st.write("**Colonnes détectées :**", ", ".join(df.columns.tolist()))
st.write("**Valeurs manquantes :**")
st.dataframe(df.isna().sum().rename("missing").reset_index().rename(columns={"index": "colonne"}), use_container_width=True)

bins_unique = cycle_summary["n_bins"].unique().tolist()
soh_constant = bool((cycle_summary["soh_nunique"] == 1).all())
st.write(f"**Bins par cycle :** {bins_unique}")
st.write(f"**Le SoH est constant dans chaque cycle :** {soh_constant}")

with st.expander("Voir un aperçu des données"):
    st.dataframe(df.head(20), use_container_width=True)

# Visualisations d'exploration
st.subheader("2. Exploration")
exp_col1, exp_col2 = st.columns(2)
with exp_col1:
    st.pyplot(plot_soh_by_battery(df))
with exp_col2:
    st.pyplot(plot_example_cycle(df))

# Split batteries
st.subheader("3. Séparation train / validation / test")
try:
    train_batteries, val_batteries, test_batteries = split_batteries(df)
except ValueError as e:
    st.error(str(e))
    st.stop()

sp1, sp2, sp3 = st.columns(3)
with sp1:
    st.write("**Train**")
    st.write(train_batteries)
with sp2:
    st.write("**Validation**")
    st.write(val_batteries)
with sp3:
    st.write("**Test**")
    st.write(test_batteries)

st.caption(
    "Ici, la séparation se fait par batteries. Cela évite qu'une même batterie se retrouve dans plusieurs ensembles, "
    "ce qui rend l'évaluation plus crédible."
)

if not train_button:
    st.info("Règle les paramètres dans la barre latérale puis clique sur **Lancer l'entraînement**.")
    st.stop()

# Préparation des données
features = ["Voltage_measured", "Current_measured", "Temperature_measured", "SoC", "cycle_number"]
target = "SoH"

with st.spinner("Création des fenêtres glissantes..."):
    X_train_raw, y_train_raw, bat_train = create_windows(df, features, target, train_batteries, window_size)
    X_val_raw, y_val_raw, bat_val = create_windows(df, features, target, val_batteries, window_size)
    X_test_raw, y_test_raw, bat_test = create_windows(df, features, target, test_batteries, window_size)

if min(len(X_train_raw), len(X_val_raw), len(X_test_raw)) == 0:
    st.error("Le nombre d'exemples générés est insuffisant. Essaie une fenêtre plus petite.")
    st.stop()

st.subheader("4. Données préparées")
dc1, dc2, dc3 = st.columns(3)
with dc1:
    st.metric("Train", X_train_raw.shape[0])
with dc2:
    st.metric("Validation", X_val_raw.shape[0])
with dc3:
    st.metric("Test", X_test_raw.shape[0])

st.write(
    f"**Forme des séquences d'entrée :** {X_train_raw.shape[1]} pas de temps × {X_train_raw.shape[2]} variables"
)

# Normalisation correcte
scaler_X = StandardScaler()
scaler_y = StandardScaler()
scaler_X.fit(X_train_raw.reshape(-1, X_train_raw.shape[-1]))
scaler_y.fit(y_train_raw.reshape(-1, 1))

X_train = transform_X(X_train_raw, scaler_X)
X_val = transform_X(X_val_raw, scaler_X)
X_test = transform_X(X_test_raw, scaler_X)

y_train = scaler_y.transform(y_train_raw.reshape(-1, 1)).flatten()
y_val = scaler_y.transform(y_val_raw.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()

# Entraînement
with st.spinner("Entraînement du modèle LSTM..."):
    model = build_model(window_size, len(features), learning_rate)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

st.subheader("5. Historique d'entraînement")
st.write(f"Nombre d'époques exécutées : {len(history.history['loss'])}")
st.pyplot(plot_training_history(history))

# Évaluation finale
with st.spinner("Évaluation finale..."):
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    baseline_pred = np.repeat(y_train_raw.mean(), len(y_true))
    baseline_mae = mean_absolute_error(y_true, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))
    baseline_r2 = r2_score(y_true, baseline_pred)

st.subheader("6. Résultats")
mt1, mt2, mt3 = st.columns(3)
with mt1:
    st.metric("MAE", f"{mae:.3f}")
with mt2:
    st.metric("RMSE", f"{rmse:.3f}")
with mt3:
    st.metric("R²", f"{r2:.4f}")

with st.expander("Comparer avec une baseline simple"):
    base_df = pd.DataFrame(
        {
            "Modèle": ["LSTM", "Baseline moyenne du train"],
            "MAE": [mae, baseline_mae],
            "RMSE": [rmse, baseline_rmse],
            "R²": [r2, baseline_r2],
        }
    )
    st.dataframe(base_df, use_container_width=True)

st.pyplot(plot_predictions(y_true, y_pred))

selected_battery = st.selectbox("Choisir une batterie du test pour voir le suivi du SoH", sorted(np.unique(bat_test)))
st.pyplot(plot_battery_tracking(y_true, y_pred, bat_test, selected_battery))

# Réponses courtes aux questions
st.subheader("7. Réponses aux questions du sujet")
st.markdown(
    """
**1. Pourquoi le SoC est-il une variable clé pour estimer le SoH ?**  
Le SoC donne le niveau de charge de la batterie au moment où les autres mesures sont observées. Il aide donc le modèle à mieux situer la tension, le courant et la température dans le cycle.

**2. Quel intérêt de découper un cycle en plusieurs fenêtres ?**  
Cela permet d'obtenir plus d'exemples d'apprentissage et de capter des motifs locaux dans les signaux.

**3. Que se passerait-il si la fenêtre était trop courte ou trop longue ?**  
Trop courte, elle contient trop peu d'information. Trop longue, elle peut ajouter du bruit et rendre l'apprentissage plus lourd.

**4. Quels risques de biais si les cycles sont mal répartis entre train et test ?**  
Le modèle peut voir des données trop proches dans les deux ensembles et donner des résultats artificiellement trop bons.

**5. Dans quels cas industriels ce type de modèle est pertinent ?**  
Ce type de modèle est utile pour les véhicules électriques, le stockage d'énergie, les systèmes solaires et les objets connectés.
"""
)

# Export prédictions
pred_df = pd.DataFrame(
    {
        "battery_id": bat_test,
        "SoH_true": y_true,
        "SoH_pred": y_pred,
        "abs_error": np.abs(y_true - y_pred),
    }
)

st.subheader("8. Télécharger les prédictions")
st.download_button(
    label="Télécharger predictions_soh_lstm.csv",
    data=pred_df.to_csv(index=False).encode("utf-8"),
    file_name="predictions_soh_lstm.csv",
    mime="text/csv",
)
