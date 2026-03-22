import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

st.set_page_config(
    page_title="Battery SoH — LSTM",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS GLOBAL ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* Fond général — force mode clair même si Streamlit est en dark mode */
.stApp, .main, section.main > div {
    background-color: #f8fafc !important;
}
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
    background-color: #f8fafc !important;
}
/* Texte par défaut lisible */
.main p, .main span, .main div, .main label, .main h1, .main h2, .main h3 {
    color: #0f172a !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f172a !important;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #0f766e !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    width: 100%;
    padding: 0.6rem 1rem;
    margin-top: 0.5rem;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #0d9488 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #334155 !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectSlider label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    color: #cbd5e1 !important;
    border-color: #334155 !important;
    background: #1e293b !important;
}

/* Metrics override */
[data-testid="metric-container"] {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.05);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #94a3b8 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 900 !important;
    color: #0f172a !important;
    letter-spacing: -1px;
}

/* Expander */
details {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 0.2rem 0.5rem !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* Dataframe */
.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── PALETTE & CHART STYLE ─────────────────────────────────────────────────────
TEAL    = "#0f766e"
BLUE    = "#2563eb"
AMBER   = "#f59e0b"
RED     = "#ef4444"
PURPLE  = "#8b5cf6"
GREEN   = "#10b981"
SLATE   = "#64748b"
INK     = "#0f172a"
PALETTE = [TEAL, BLUE, AMBER, RED, PURPLE, GREEN, "#ec4899", "#84cc16"]

plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#f8fafc",
    "axes.edgecolor":   "#e2e8f0",
    "axes.grid":        True,
    "grid.color":       "#e2e8f0",
    "grid.linewidth":   0.8,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.titlecolor":  INK,
    "axes.labelcolor":  SLATE,
    "xtick.color":      SLATE,
    "ytick.color":      SLATE,
    "legend.frameon":   False,
    "legend.fontsize":  10,
})


# ── FONCTIONS UTILITAIRES ─────────────────────────────────────────────────────
def section(title, subtitle=""):
    st.markdown(f"""
    <div style="margin:2rem 0 1rem;padding-bottom:10px;border-bottom:2px solid #e2e8f0;background:transparent;">
        <span style="font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:2px;color:{TEAL};">
            {'● ' + subtitle if subtitle else ''}
        </span>
        <h3 style="margin:4px 0 0;font-size:18px;font-weight:800;color:#0f172a;letter-spacing:-.5px;">{title}</h3>
    </div>
    """, unsafe_allow_html=True)


def kpi_card(label, value, color=TEAL, icon=""):
    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;padding:16px 20px;
                border-top:3px solid {color};box-shadow:0 1px 4px rgba(0,0,0,.04);">
        <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;
                    color:#64748b;margin-bottom:8px;">{icon} {label}</div>
        <div style="font-size:24px;font-weight:900;color:#0f172a;letter-spacing:-1px;">{value}</div>
    </div>
    """, unsafe_allow_html=True)


REQUIRED_COLUMNS = [
    "Voltage_measured", "Current_measured", "Temperature_measured",
    "SoC", "cycle_number", "battery_id", "SoH",
]


@st.cache_data(max_entries=1)
def load_dataset_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(max_entries=1)
def load_dataset_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def check_columns(df: pd.DataFrame):
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


@st.cache_data(max_entries=1)
def prepare_cycle_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["battery_id", "cycle_number"])
        .agg(n_bins=("SoH","size"), soh_nunique=("SoH","nunique"), soh=("SoH","first"))
        .reset_index()
    )


def split_batteries(df: pd.DataFrame):
    all_batteries = sorted(df["battery_id"].unique())
    if len(all_batteries) < 9:
        raise ValueError("Il faut au moins 9 batteries pour ce split.")
    test_batteries  = all_batteries[-4:]
    val_batteries   = all_batteries[-8:-4]
    train_batteries = all_batteries[:-8]
    return train_batteries, val_batteries, test_batteries


@st.cache_data(max_entries=1)
def create_windows(dataframe, feature_cols, target_col, battery_list, window_size):
    """Une seule fenêtre par cycle (début) pour limiter la mémoire."""
    subset = dataframe[dataframe["battery_id"].isin(battery_list)].copy()
    subset = subset.sort_values(["battery_id","cycle_number","SoC"], ascending=[True,True,False])
    X_list, y_list, battery_ids = [], [], []
    for (bat_id, _), group in subset.groupby(["battery_id","cycle_number"]):
        values = group[feature_cols].to_numpy(dtype=np.float32)
        soh = float(group[target_col].iloc[0])
        if len(values) < window_size:
            continue
        # Une seule fenêtre par cycle au lieu de toutes les fenêtres glissantes
        X_list.append(values[:window_size])
        y_list.append(soh)
        battery_ids.append(bat_id)
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            np.array(battery_ids))


def transform_X(X, scaler):
    shape = X.shape
    return scaler.transform(X.reshape(-1, shape[-1])).reshape(shape)


def build_model(window_size, n_features, learning_rate):
    model = keras.Sequential([
        layers.Input(shape=(window_size, n_features)),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(16),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mse", metrics=["mae"])
    return model


# ── GRAPHIQUES ────────────────────────────────────────────────────────────────
def plot_soh_by_battery(df):
    soh_cycle = df.groupby(["battery_id","cycle_number"])["SoH"].first().reset_index()
    batteries = sorted(df["battery_id"].unique())[:4]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, bat in enumerate(batteries):
        d = soh_cycle[soh_cycle["battery_id"] == bat]
        ax.plot(d["cycle_number"], d["SoH"], label=str(bat),
                linewidth=2.5, color=PALETTE[i])
    ax.axhline(80, linestyle="--", linewidth=1.5, color=RED, alpha=0.7, label="Seuil 80%")
    ax.fill_between(ax.get_xlim(), 0, 80, alpha=0.04, color=RED)
    ax.set_title("Évolution du SoH par batterie")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SoH (%)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_example_cycle(df):
    bat = sorted(df["battery_id"].unique())[0]
    cyc = int(df[df["battery_id"] == bat]["cycle_number"].min())
    data = df[(df["battery_id"] == bat) & (df["cycle_number"] == cyc)].copy()
    cols   = ["Voltage_measured","Current_measured","Temperature_measured","SoC"]
    titles = ["Tension (V)","Courant (A)","Température (°C)","SoC (%)"]
    colors = [TEAL, BLUE, AMBER, PURPLE]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5))
    for ax, col, title, color in zip(axes.ravel(), cols, titles, colors):
        ax.plot(data[col].values, linewidth=2, color=color)
        ax.fill_between(range(len(data)), data[col].values, alpha=0.08, color=color)
        ax.set_title(title)
        ax.set_xlabel("Bin")
    fig.suptitle(f"Batterie {bat} — cycle {cyc}", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"],     label="Train",      linewidth=2.5, color=TEAL)
    axes[0].plot(history.history["val_loss"], label="Validation", linewidth=2.5, color=AMBER, linestyle="--")
    axes[0].set_title("Évolution de la Loss (MSE)")
    axes[0].set_xlabel("Époque")
    axes[0].legend()
    axes[1].plot(history.history["mae"],      label="Train",      linewidth=2.5, color=TEAL)
    axes[1].plot(history.history["val_mae"],  label="Validation", linewidth=2.5, color=AMBER, linestyle="--")
    axes[1].set_title("Évolution de la MAE")
    axes[1].set_xlabel("Époque")
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_predictions(y_true, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_true, y_pred, alpha=0.35, s=10, color=TEAL)
    lims = [min(y_true.min(), y_pred.min())-2, max(y_true.max(), y_pred.max())+2]
    axes[0].plot(lims, lims, "--", linewidth=2, color=AMBER)
    axes[0].set_title("SoH réel vs SoH prédit")
    axes[0].set_xlabel("SoH réel")
    axes[0].set_ylabel("SoH prédit")

    errors = y_pred - y_true
    axes[1].hist(errors, bins=40, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.5)
    axes[1].axvline(0, linestyle="--", linewidth=2, color=RED)
    axes[1].set_title("Distribution des erreurs")
    axes[1].set_xlabel("Erreur (prédit − réel)")
    axes[1].set_ylabel("Fréquence")
    fig.tight_layout()
    return fig


def plot_battery_tracking(y_true, y_pred, bat_test, selected_battery):
    mask = bat_test == selected_battery
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true[mask], label="SoH réel",   linewidth=2.5, color=TEAL)
    ax.plot(y_pred[mask], label="SoH prédit", linewidth=2.5, color=AMBER, linestyle="--")
    ax.fill_between(range(mask.sum()), y_true[mask], y_pred[mask], alpha=0.1, color=AMBER)
    ax.axhline(80, linestyle=":", linewidth=1.5, color=RED, alpha=0.6)
    ax.set_title(f"Suivi du SoH — batterie {selected_battery}")
    ax.set_xlabel("Fenêtres")
    ax.set_ylabel("SoH (%)")
    ax.legend()
    fig.tight_layout()
    return fig


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,{INK} 0%,#1e3a5f 55%,{TEAL} 100%);
            padding:2rem 2.5rem;border-radius:20px;margin-bottom:1.5rem;">
    <div style="font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:3px;
                color:#99f6e4;margin-bottom:12px;">Deep Learning · Prédiction SoH</div>
    <h1 style="font-size:32px;font-weight:900;color:#fff;margin:0 0 10px;letter-spacing:-1px;">
        🔋 Battery Health Prediction
    </h1>
    <p style="font-size:14px;color:#94a3b8;margin:0;line-height:1.7;">
        Modèle <strong style="color:#e2e8f0;">LSTM</strong> pour prédire l'état de santé
        des batteries lithium-ion à partir de cycles de charge/décharge.
    </p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:1rem 0 0.5rem;">
        <div style="font-size:10px;font-weight:800;text-transform:uppercase;
                    letter-spacing:2px;color:#0f766e;margin-bottom:4px;">SoH · LSTM</div>
        <div style="font-size:17px;font-weight:800;color:#fff;">Configuration</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#94a3b8;margin-bottom:4px;">Données</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Fichier CSV", type=["csv"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<p style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#94a3b8;margin-bottom:4px;">Hyperparamètres</p>', unsafe_allow_html=True)
    window_size   = st.slider("Fenêtre glissante", min_value=3,  max_value=15,  value=10)
    epochs        = st.slider("Époques max",       min_value=5,  max_value=60,  value=30)
    batch_size    = st.select_slider("Batch size", options=[16,32,64,128], value=64)
    learning_rate = st.select_slider("Learning rate", options=[0.0005,0.001,0.002,0.005], value=0.001)
    patience      = st.slider("Early stopping patience", min_value=3, max_value=15, value=8)

    st.markdown("---")
    train_button = st.button("🚀 Lancer l'entraînement")

# ── CHARGEMENT ────────────────────────────────────────────────────────────────
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
        st.markdown(f"""
        <div style="background:#f0fdfa;border:1px solid #99f6e4;border-radius:14px;padding:20px 24px;">
            <div style="font-size:15px;font-weight:700;color:{TEAL};margin-bottom:8px;">📂 Aucune donnée détectée</div>
            <p style="font-size:13px;color:#475569;margin:0;">
                Importe ton fichier <strong>battery_health_dataset.csv</strong> depuis la barre latérale gauche,
                ou place-le à la racine du projet.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    df = load_dataset_from_path(str(existing[0]))
    dataset_label = existing[0].name

missing = check_columns(df)
if missing:
    st.error(f"Colonnes manquantes : {missing}")
    st.stop()

if tf is None:
    st.error(f"TensorFlow non disponible. Détail : {TF_IMPORT_ERROR}")
    st.stop()

np.random.seed(42)
tf.random.set_seed(42)
df = df.copy()

st.markdown(f"""
<div style="background:#ecfdf5;border:1px solid #6ee7b7;border-radius:10px;
            padding:10px 16px;font-size:13px;font-weight:600;color:#065f46;margin-bottom:1rem;">
    ✓ Données chargées — <strong>{dataset_label}</strong>
</div>
""", unsafe_allow_html=True)

# ── SECTION 1 — APERÇU ───────────────────────────────────────────────────────
section("Vérification de la base", "Étape 1")
cycle_summary = prepare_cycle_summary(df)

c1, c2, c3, c4 = st.columns(4)
with c1: kpi_card("Observations", f"{df.shape[0]:,}".replace(",", " "), TEAL, "📊")
with c2: kpi_card("Variables", df.shape[1], BLUE, "📋")
with c3: kpi_card("Batteries", df["battery_id"].nunique(), PURPLE, "🔋")
with c4: kpi_card("Cycles", cycle_summary.shape[0], AMBER, "🔄")

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

col_left, col_right = st.columns(2)
with col_left:
    st.markdown("**Colonnes détectées**")
    st.code(", ".join(df.columns.tolist()), language=None)
with col_right:
    st.markdown("**Valeurs manquantes**")
    nan_df = df.isna().sum().rename("Manquantes").reset_index().rename(columns={"index":"Colonne"})
    st.dataframe(nan_df, use_container_width=True, hide_index=True)

bins_unique  = cycle_summary["n_bins"].unique().tolist()
soh_constant = bool((cycle_summary["soh_nunique"] == 1).all())
st.markdown(f"""
<div style="display:flex;gap:12px;margin-top:0.5rem;">
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
              padding:10px 16px;font-size:12px;color:#475569;">
    <strong>Bins par cycle :</strong> {bins_unique}
  </div>
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
              padding:10px 16px;font-size:12px;color:#475569;">
    <strong>SoH constant par cycle :</strong> {'✓ Oui' if soh_constant else '✗ Non'}
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("Aperçu des 20 premières lignes"):
    st.dataframe(df.head(20), use_container_width=True)

# ── SECTION 2 — EXPLORATION ──────────────────────────────────────────────────
section("Exploration des données", "Étape 2")
exp_c1, exp_c2 = st.columns(2)
with exp_c1:
    st.pyplot(plot_soh_by_battery(df))
with exp_c2:
    st.pyplot(plot_example_cycle(df))

# ── SECTION 3 — SPLIT ────────────────────────────────────────────────────────
section("Séparation Train / Validation / Test", "Étape 3")
try:
    train_batteries, val_batteries, test_batteries = split_batteries(df)
except ValueError as e:
    st.error(str(e))
    st.stop()

sp1, sp2, sp3 = st.columns(3)
for col, label, batteries, color in [
    (sp1, "Train",      train_batteries, TEAL),
    (sp2, "Validation", val_batteries,   BLUE),
    (sp3, "Test",       test_batteries,  AMBER),
]:
    with col:
        st.markdown(f"""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;
                    padding:14px 16px;border-top:3px solid {color};">
            <div style="font-size:10px;font-weight:800;text-transform:uppercase;
                        letter-spacing:1px;color:{color};margin-bottom:8px;">{label}</div>
            <div style="font-size:13px;color:#475569;">{', '.join(str(b) for b in batteries)}</div>
            <div style="font-size:11px;color:#94a3b8;margin-top:6px;">{len(batteries)} batteries</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
st.caption("La séparation se fait par batterie pour éviter toute fuite de données entre les ensembles.")

if not train_button:
    st.markdown(f"""
    <div style="background:#f8fafc;border:1px dashed #cbd5e1;border-radius:14px;
                padding:24px;text-align:center;margin-top:1rem;">
        <div style="font-size:24px;margin-bottom:8px;">🚀</div>
        <div style="font-size:15px;font-weight:700;color:{INK};margin-bottom:4px;">Prêt à entraîner</div>
        <div style="font-size:13px;color:#64748b;">Configure les paramètres dans la barre latérale puis clique sur <strong>Lancer l'entraînement</strong>.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── SECTION 4 — PRÉPARATION ───────────────────────────────────────────────────
features = ["Voltage_measured","Current_measured","Temperature_measured","SoC","cycle_number"]
target   = "SoH"

with st.spinner("Création des fenêtres glissantes..."):
    X_train_raw, y_train_raw, bat_train = create_windows(df, features, target, train_batteries, window_size)
    X_val_raw,   y_val_raw,   bat_val   = create_windows(df, features, target, val_batteries,   window_size)
    X_test_raw,  y_test_raw,  bat_test  = create_windows(df, features, target, test_batteries,  window_size)

if min(len(X_train_raw), len(X_val_raw), len(X_test_raw)) == 0:
    st.error("Exemples insuffisants. Réduis la taille de la fenêtre.")
    st.stop()

section("Données préparées", "Étape 4")
dc1, dc2, dc3 = st.columns(3)
with dc1: kpi_card("Séquences train",      f"{X_train_raw.shape[0]:,}", TEAL,   "📦")
with dc2: kpi_card("Séquences validation", f"{X_val_raw.shape[0]:,}",   BLUE,   "📦")
with dc3: kpi_card("Séquences test",       f"{X_test_raw.shape[0]:,}",  PURPLE, "📦")

st.markdown(f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
            padding:10px 16px;font-size:13px;color:#475569;margin-top:0.5rem;">
    Forme des séquences d'entrée : <strong>{X_train_raw.shape[1]}</strong> pas de temps
    × <strong>{X_train_raw.shape[2]}</strong> variables
</div>
""", unsafe_allow_html=True)

# Normalisation
scaler_X = StandardScaler()
scaler_y = StandardScaler()
scaler_X.fit(X_train_raw.reshape(-1, X_train_raw.shape[-1]))
scaler_y.fit(y_train_raw.reshape(-1, 1))

X_train = transform_X(X_train_raw, scaler_X)
X_val   = transform_X(X_val_raw,   scaler_X)
X_test  = transform_X(X_test_raw,  scaler_X)
y_train = scaler_y.transform(y_train_raw.reshape(-1,1)).flatten()
y_val   = scaler_y.transform(y_val_raw.reshape(-1,1)).flatten()
y_test  = scaler_y.transform(y_test_raw.reshape(-1,1)).flatten()
y_train_mean = float(y_train_raw.mean())
# Libérer la mémoire des arrays bruts
del X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw

# ── SECTION 5 — ENTRAÎNEMENT ─────────────────────────────────────────────────
section("Entraînement du modèle LSTM", "Étape 5")

with st.spinner("Entraînement en cours..."):
    model = build_model(window_size, len(features), learning_rate)
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stop], verbose=0,
    )

st.markdown(f"""
<div style="background:#ecfdf5;border:1px solid #6ee7b7;border-radius:10px;
            padding:10px 16px;font-size:13px;font-weight:600;color:#065f46;margin-bottom:1rem;">
    ✓ Entraînement terminé — <strong>{len(history.history['loss'])}</strong> époques exécutées
</div>
""", unsafe_allow_html=True)
st.pyplot(plot_training_history(history))

# ── SECTION 6 — RÉSULTATS ────────────────────────────────────────────────────
with st.spinner("Évaluation finale..."):
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    r2     = r2_score(y_true, y_pred)
    baseline_pred = np.repeat(y_train_mean, len(y_true))
    baseline_mae  = mean_absolute_error(y_true, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))
    baseline_r2   = r2_score(y_true, baseline_pred)

section("Résultats du modèle", "Étape 6")
m1, m2, m3 = st.columns(3)
r2_color = GREEN if r2 >= 0.90 else (AMBER if r2 >= 0.75 else RED)
with m1: kpi_card("MAE",  f"{mae:.4f}",  GREEN if mae < 2 else AMBER, "📉")
with m2: kpi_card("RMSE", f"{rmse:.4f}", GREEN if rmse < 3 else AMBER, "📐")
with m3: kpi_card("R²",   f"{r2:.4f}",  r2_color, "🎯")

with st.expander("Comparer avec la baseline (moyenne train)"):
    base_df = pd.DataFrame({
        "Modèle": ["LSTM", "Baseline"],
        "MAE":  [round(mae,4),  round(baseline_mae,4)],
        "RMSE": [round(rmse,4), round(baseline_rmse,4)],
        "R²":   [round(r2,4),   round(baseline_r2,4)],
    })
    st.dataframe(base_df, use_container_width=True, hide_index=True)

st.pyplot(plot_predictions(y_true, y_pred))

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
selected_battery = st.selectbox(
    "Choisir une batterie du test pour voir le suivi du SoH",
    sorted(np.unique(bat_test))
)
st.pyplot(plot_battery_tracking(y_true, y_pred, bat_test, selected_battery))

# ── SECTION 7 — QUESTIONS ────────────────────────────────────────────────────
section("Réponses aux questions du sujet", "Étape 7")
questions = [
    ("Pourquoi le SoC est-il une variable clé pour estimer le SoH ?",
     "Le SoC indique le niveau de charge instantané de la batterie. Il aide le modèle à contextualiser "
     "les mesures de tension, courant et température dans le cycle, améliorant ainsi la précision."),
    ("Quel intérêt de découper un cycle en plusieurs fenêtres ?",
     "Cela multiplie les exemples d'entraînement et permet au LSTM de capter des motifs locaux "
     "dans les signaux mesurés à différents stades du cycle."),
    ("Que se passerait-il si la fenêtre était trop courte ou trop longue ?",
     "Trop courte : information insuffisante, le modèle manque de contexte. "
     "Trop longue : bruit accru, entraînement plus lourd, risque de sur-apprentissage."),
    ("Quels risques de biais si les cycles sont mal répartis entre train et test ?",
     "Le modèle verrait des données similaires dans les deux ensembles, produisant "
     "des métriques artificiellement bonnes sans généralisation réelle."),
    ("Dans quels cas industriels ce type de modèle est pertinent ?",
     "Véhicules électriques, stockage d'énergie renouvelable, systèmes de backup, "
     "objets connectés (IoT) — partout où surveiller la dégradation de la batterie est critique."),
]
for i, (q, a) in enumerate(questions, 1):
    with st.expander(f"Q{i}. {q}"):
        st.markdown(f"""
        <div style="font-size:13px;color:#475569;line-height:1.8;padding:4px 0;">
            {a}
        </div>
        """, unsafe_allow_html=True)

# ── SECTION 8 — EXPORT ───────────────────────────────────────────────────────
section("Téléchargement des prédictions", "Étape 8")
pred_df = pd.DataFrame({
    "battery_id": bat_test,
    "SoH_true":   y_true,
    "SoH_pred":   y_pred,
    "abs_error":  np.abs(y_true - y_pred),
})
c_dl, c_info = st.columns([1,2])
with c_dl:
    st.download_button(
        label="📥 Télécharger predictions_soh_lstm.csv",
        data=pred_df.to_csv(index=False).encode("utf-8"),
        file_name="predictions_soh_lstm.csv",
        mime="text/csv",
        use_container_width=True,
    )
with c_info:
    st.markdown(f"""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                padding:12px 16px;font-size:12px;color:#64748b;">
        <strong>{len(pred_df)}</strong> prédictions sur
        <strong>{pred_df['battery_id'].nunique()}</strong> batteries de test.<br>
        MAE moyenne : <strong>{pred_df['abs_error'].mean():.4f}</strong>
    </div>
    """, unsafe_allow_html=True)
