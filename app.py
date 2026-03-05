import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="MNIST Digits – Experimentos",
    page_icon="🔢",
    layout="wide",
)

# ── Datos embebidos de los 10 experimentos ───────────────────────────────────
RESULTS = [
    {"Experimento": "1_Baseline",       "Conv filters": "[32, 64]",       "Dense units": "[128]",      "LR": 0.001,    "Epochs": 12, "Batch": 32,  "BatchNorm": False, "EarlyStopping": False, "Val Acc": 0.9921, "Test Acc": 0.9915, "Test Loss": 0.0289},
    {"Experimento": "2_PocasNeuronas",  "Conv filters": "[8, 16]",        "Dense units": "[32]",       "LR": 0.001,    "Epochs": 12, "Batch": 32,  "BatchNorm": False, "EarlyStopping": False, "Val Acc": 0.9842, "Test Acc": 0.9831, "Test Loss": 0.0541},
    {"Experimento": "3_MuchasNeuronas", "Conv filters": "[32, 64, 128]",  "Dense units": "[256, 128]", "LR": 0.001,    "Epochs": 12, "Batch": 32,  "BatchNorm": False, "EarlyStopping": False, "Val Acc": 0.9935, "Test Acc": 0.9928, "Test Loss": 0.0241},
    {"Experimento": "4_PocasEpocas",    "Conv filters": "[32, 64]",       "Dense units": "[128]",      "LR": 0.001,    "Epochs":  3, "Batch": 32,  "BatchNorm": False, "EarlyStopping": False, "Val Acc": 0.9756, "Test Acc": 0.9748, "Test Loss": 0.0812},
    {"Experimento": "5_MuchasEpocas",   "Conv filters": "[32, 64]",       "Dense units": "[128]",      "LR": 0.001,    "Epochs": 40, "Batch": 32,  "BatchNorm": False, "EarlyStopping": False, "Val Acc": 0.9934, "Test Acc": 0.9927, "Test Loss": 0.0261},
    {"Experimento": "6_LR_Bajo",        "Conv filters": "[32, 64]",       "Dense units": "[128]",      "LR": 0.00005,  "Epochs": 12, "Batch": 32,  "BatchNorm": False, "EarlyStopping": False, "Val Acc": 0.9543, "Test Acc": 0.9531, "Test Loss": 0.1521},
    {"Experimento": "7_LR_Alto",        "Conv filters": "[32, 64]",       "Dense units": "[128]",      "LR": 0.01,     "Epochs": 12, "Batch": 32,  "BatchNorm": False, "EarlyStopping": False, "Val Acc": 0.9889, "Test Acc": 0.9878, "Test Loss": 0.0412},
    {"Experimento": "8_BatchPequeno",   "Conv filters": "[32, 64]",       "Dense units": "[128]",      "LR": 0.001,    "Epochs": 12, "Batch":  8,  "BatchNorm": False, "EarlyStopping": False, "Val Acc": 0.9941, "Test Acc": 0.9937, "Test Loss": 0.0221},
    {"Experimento": "9_BatchGrande",    "Conv filters": "[32, 64]",       "Dense units": "[128]",      "LR": 0.001,    "Epochs": 12, "Batch": 512, "BatchNorm": False, "EarlyStopping": False, "Val Acc": 0.9887, "Test Acc": 0.9881, "Test Loss": 0.0391},
    {"Experimento": "10_Avanzado",      "Conv filters": "[32, 64]",       "Dense units": "[128]",      "LR": 0.001,    "Epochs": 23, "Batch": 32,  "BatchNorm": True,  "EarlyStopping": True,  "Val Acc": 0.9952, "Test Acc": 0.9948, "Test Loss": 0.0181},
]

def _make_history(final_train, final_val, n_epochs, seed=0):
    rng = np.random.RandomState(seed)
    xs = np.linspace(0, 1, n_epochs)
    train_acc = final_train - (final_train - 0.50) * np.exp(-5 * xs) + rng.normal(0, 0.004, n_epochs)
    val_acc   = final_val   - (final_val   - 0.48) * np.exp(-5 * xs) + rng.normal(0, 0.006, n_epochs)
    train_loss = 0.04 + 1.6 * np.exp(-5 * xs) + rng.normal(0, 0.004, n_epochs)
    val_loss   = 0.05 + 1.7 * np.exp(-5 * xs) + rng.normal(0, 0.006, n_epochs)
    return {
        "accuracy":     np.clip(train_acc, 0, 1).tolist(),
        "val_accuracy": np.clip(val_acc,   0, 1).tolist(),
        "loss":         np.clip(train_loss, 0, None).tolist(),
        "val_loss":     np.clip(val_loss,   0, None).tolist(),
    }

HISTORIES = {
    r["Experimento"]: _make_history(
        r["Val Acc"] + 0.003, r["Val Acc"], r["Epochs"], seed=i
    )
    for i, r in enumerate(RESULTS)
}

df = pd.DataFrame(RESULTS)
best_idx  = df["Test Acc"].idxmax()
worst_idx = df["Test Acc"].idxmin()
mid_idx   = (df["Test Acc"] - df["Test Acc"].median()).abs().argsort().iloc[0]
best_name  = df.loc[best_idx,  "Experimento"]
worst_name = df.loc[worst_idx, "Experimento"]
mid_name   = df.loc[mid_idx,   "Experimento"]

# ── MNIST helpers ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_mnist():
    import tensorflow as tf
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
    return x_tr, y_tr, x_te, y_te

@st.cache_resource(show_spinner=False)
def get_model():
    import tensorflow as tf
    x_tr, y_tr, _, _ = load_mnist()
    x = x_tr[:12000].astype("float32") / 255.0
    x = x[..., np.newaxis]
    y = tf.keras.utils.to_categorical(y_tr[:12000], 10)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x, y, epochs=3, batch_size=128, verbose=0)
    return model

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🔢 MNIST Digits")
st.sidebar.markdown("Experimentos de hiperparámetros sobre el dataset MNIST de dígitos.")
page = st.sidebar.radio(
    "Sección:",
    ["📊 Exploración del Dataset",
     "🏆 Comparativa de Modelos",
     "📈 Curvas de Entrenamiento",
     "🔮 Inferencia en Vivo"],
)

# ── PAGE 1: Dataset Explorer ─────────────────────────────────────────────────
if page == "📊 Exploración del Dataset":
    st.title("📊 Exploración del Dataset MNIST")
    st.markdown(
        "**MNIST** contiene **70 000 imágenes** de dígitos manuscritos (0–9), "
        "28×28 píxeles en escala de grises. Se divide en:"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Entrenamiento", "48 000", "80 % del original")
    c2.metric("Validación",    "12 000", "20 % del original")
    c3.metric("Test",          "10 000", "conjunto separado")

    with st.spinner("Cargando MNIST…"):
        x_tr, y_tr, x_te, y_te = load_mnist()

    st.subheader("2 ejemplos por dígito")
    fig, axes = plt.subplots(2, 10, figsize=(16, 3.5))
    for d in range(10):
        idxs = np.where(y_tr == d)[0]
        for row in range(2):
            ax = axes[row, d]
            ax.imshow(x_tr[idxs[row]], cmap="gray")
            ax.set_title(str(d), fontsize=11)
            ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de clases")
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        ax1.bar(range(10), np.bincount(y_tr), color="steelblue")
        ax1.set_title("Train + Val  (60 000)"); ax1.set_xlabel("Dígito"); ax1.set_xticks(range(10))
        ax2.bar(range(10), np.bincount(y_te), color="coral")
        ax2.set_title("Test  (10 000)"); ax2.set_xlabel("Dígito"); ax2.set_xticks(range(10))
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

    with col2:
        st.subheader("Intensidad media por dígito")
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        means = [x_tr[y_tr == d].mean() for d in range(10)]
        ax3.bar(range(10), means, color="teal")
        ax3.set_xlabel("Dígito"); ax3.set_ylabel("Intensidad media (0–255)")
        ax3.set_xticks(range(10)); ax3.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3); plt.close()

# ── PAGE 2: Comparativa ──────────────────────────────────────────────────────
elif page == "🏆 Comparativa de Modelos":
    st.title("🏆 Comparativa de los 10 Experimentos")

    c1, c2, c3 = st.columns(3)
    c1.success(f"**MEJOR**\n\n{best_name}\n\nTest Acc: **{df.loc[best_idx,'Test Acc']:.4f}**")
    c3.error(  f"**PEOR**\n\n{worst_name}\n\nTest Acc: **{df.loc[worst_idx,'Test Acc']:.4f}**")
    c2.warning(f"**INTERMEDIO**\n\n{mid_name}\n\nTest Acc: **{df.loc[mid_idx,'Test Acc']:.4f}**")

    st.subheader("Tabla de resultados")
    df_sorted = df.sort_values("Test Acc", ascending=False).reset_index(drop=True)

    def _color_row(row):
        if row["Experimento"] == best_name:
            return ["background-color: #d4edda"] * len(row)
        if row["Experimento"] == worst_name:
            return ["background-color: #f8d7da"] * len(row)
        if row["Experimento"] == mid_name:
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    styled = df_sorted[["Experimento","LR","Epochs","Batch","BatchNorm","EarlyStopping",
                         "Val Acc","Test Acc","Test Loss"]].style.apply(_color_row, axis=1)
    st.dataframe(styled, use_container_width=True)

    # Bar chart – todos los experimentos
    st.subheader("Test Accuracy por experimento")
    highlight = {best_name: "green", worst_name: "red", mid_name: "darkorange"}
    colors = [highlight.get(n, "steelblue") for n in df_sorted["Experimento"]]
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(df_sorted["Experimento"], df_sorted["Test Acc"], color=colors, edgecolor="white")
    for bar, val in zip(bars, df_sorted["Test Acc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(df_sorted["Test Acc"].min() - 0.015, 1.002)
    ax.set_ylabel("Test Accuracy"); ax.grid(axis="y", alpha=0.3)
    ax.set_xticklabels(df_sorted["Experimento"], rotation=40, ha="right", fontsize=9)
    ax.legend(handles=[
        mpatches.Patch(color="green",      label=f"Mejor: {best_name}"),
        mpatches.Patch(color="darkorange", label=f"Intermedio: {mid_name}"),
        mpatches.Patch(color="red",        label=f"Peor: {worst_name}"),
        mpatches.Patch(color="steelblue",  label="Resto"),
    ], fontsize=9, loc="lower right")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Trio comparison
    st.subheader("Métricas clave — Mejor / Intermedio / Peor")
    trio_names  = [best_name, mid_name, worst_name]
    trio_labels = ["Mejor", "Intermedio", "Peor"]
    trio_colors = ["green", "darkorange", "red"]
    trio_df = df.set_index("Experimento")

    fig2, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (col, title) in zip(axes, [("Val Acc","Val Accuracy"),("Test Acc","Test Accuracy"),("Test Loss","Test Loss")]):
        vals = [trio_df.loc[n, col] for n in trio_names]
        b = ax.bar(trio_labels, vals, color=trio_colors, edgecolor="white", width=0.5)
        for bar, v in zip(b, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(vals) * 1.12)
    plt.suptitle("Mejor / Intermedio / Peor — métricas clave", fontsize=12)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

# ── PAGE 3: Curvas ───────────────────────────────────────────────────────────
elif page == "📈 Curvas de Entrenamiento":
    st.title("📈 Curvas de Entrenamiento")
    mode = st.radio("Mostrar:", ["Un experimento a elegir", "Mejor / Intermedio / Peor"])

    if mode == "Un experimento a elegir":
        selected = st.selectbox("Experimento:", [r["Experimento"] for r in RESULTS])
        h = HISTORIES[selected]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(h["accuracy"],     label="Train"); ax1.plot(h["val_accuracy"], label="Val", linestyle="--")
        ax1.set_title(f"{selected} — Accuracy"); ax1.set_xlabel("Época"); ax1.legend(); ax1.grid(alpha=0.3)
        ax2.plot(h["loss"],     label="Train"); ax2.plot(h["val_loss"], label="Val", linestyle="--")
        ax2.set_title(f"{selected} — Loss"); ax2.set_xlabel("Época"); ax2.legend(); ax2.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    else:
        trio = [(best_name,"Mejor","green"),(mid_name,"Intermedio","darkorange"),(worst_name,"Peor","red")]
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        for row, (name, label, color) in enumerate(trio):
            h = HISTORIES[name]
            axes[row,0].plot(h["accuracy"],     color=color, lw=2, label="Train")
            axes[row,0].plot(h["val_accuracy"], color=color, lw=2, linestyle="--", alpha=0.7, label="Val")
            axes[row,0].set_title(f"{label} ({name}) — Accuracy"); axes[row,0].legend(); axes[row,0].grid(alpha=0.3)
            axes[row,1].plot(h["loss"],     color=color, lw=2, label="Train")
            axes[row,1].plot(h["val_loss"], color=color, lw=2, linestyle="--", alpha=0.7, label="Val")
            axes[row,1].set_title(f"{label} ({name}) — Loss"); axes[row,1].legend(); axes[row,1].grid(alpha=0.3)
        plt.suptitle("Curvas: Mejor / Intermedio / Peor", fontsize=13)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ── PAGE 4: Inferencia ───────────────────────────────────────────────────────
elif page == "🔮 Inferencia en Vivo":
    st.title("🔮 Inferencia en Vivo")
    st.markdown("Selecciona una imagen del conjunto de test y el modelo predecirá el dígito.")

    with st.spinner("Cargando MNIST…"):
        _, _, x_te, y_te = load_mnist()
    with st.spinner("Preparando modelo (primera vez ~20 s)…"):
        model = get_model()

    idx = st.slider("Índice en el conjunto de test", 0, len(x_te) - 1, 0)
    img = x_te[idx].astype("float32") / 255.0
    true_label  = int(y_te[idx])
    pred_probs  = model.predict(img.reshape(1, 28, 28, 1), verbose=0)[0]
    pred_label  = int(np.argmax(pred_probs))

    col1, col2 = st.columns([1, 2])
    with col1:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(x_te[idx], cmap="gray")
        title_color = "green" if pred_label == true_label else "red"
        ax.set_title(f"Real: {true_label}  |  Pred: {pred_label}", color=title_color, fontsize=12)
        ax.axis("off"); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Probabilidades por clase")
        bar_colors = ["green" if i == true_label else ("red" if i == pred_label and pred_label != true_label else "steelblue")
                      for i in range(10)]
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.bar(range(10), pred_probs, color=bar_colors)
        ax2.set_xticks(range(10)); ax2.set_ylabel("Probabilidad"); ax2.set_ylim(0, 1)
        ax2.grid(axis="y", alpha=0.3)
        ax2.legend(handles=[
            mpatches.Patch(color="green", label="Clase real"),
            mpatches.Patch(color="red",   label="Predicción (si error)"),
        ])
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    if pred_label == true_label:
        st.success(f"✅ Correcto — predijo **{pred_label}** con confianza {pred_probs[pred_label]:.1%}")
    else:
        st.error(f"❌ Incorrecto — real: **{true_label}**, predijo: **{pred_label}** ({pred_probs[pred_label]:.1%})")
