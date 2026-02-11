import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

# ================= CONFIG =================
IMG_SIZE = 224
MODEL_KERAS = "models/best_model.keras"
MODEL_SAVED = "models/deepfashion_mobilenetv2_savedmodel"
LABELS_JSON = "data/processed/labels.json"
ATTRIBUTES_TXT = "anno/list_attr_cloth.txt"

# ================= STREAMLIT CONFIG =================
st.set_page_config(
    page_title="👗 DeepFashion App",
    layout="wide"
)

st.title("👗 DeepFashion – Clasificación de Prendas")
st.markdown("Clasificación automática de prendas y detección de atributos visuales.")

# ================= UTILIDADES =================
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_SAVED):
        model = tf.keras.models.load_model(MODEL_SAVED, compile=False)
    elif os.path.exists(MODEL_KERAS):
        model = tf.keras.models.load_model(MODEL_KERAS, compile=False)
        os.makedirs(MODEL_SAVED, exist_ok=True)
        model.save(MODEL_SAVED)
    else:
        st.error("❌ No se encontró el modelo entrenado.")
        st.stop()
    return model

@st.cache_data
def load_labels():
    if os.path.exists(LABELS_JSON):
        with open(LABELS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_data
def load_attributes():
    if os.path.exists(ATTRIBUTES_TXT):
        with open(ATTRIBUTES_TXT, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()][2:]
        return [ln.split()[0] for ln in lines]
    return None

def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img).astype("float32")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    return np.expand_dims(x, axis=0)

def explain_attr(name, prob):
    if prob > 0.75:
        return "Muy probable"
    elif prob > 0.5:
        return "Probable"
    elif prob > 0.25:
        return "Poco probable"
    else:
        return "Improbable"

# ================= CARGA MODELO =================
model = load_model()
labels = load_labels()
attr_names = load_attributes()

# ================= SIDEBAR – VARIABLES (COPIADAS) =================
st.sidebar.header("Variables del Producto")

# NUMÉRICAS
product_like_count = st.sidebar.number_input(
    "Cantidad de Likes", min_value=0, value=1
)

seller_price = st.sidebar.number_input(
    "Precio del vendedor", min_value=0.0, value=50.0
)

buyers_fees = st.sidebar.number_input(
    "Comisiones del comprador", min_value=0.0, value=5.0
)

seller_products_sold = st.sidebar.number_input(
    "Productos vendidos por el vendedor", min_value=0, value=10
)

seller_num_followers = st.sidebar.number_input(
    "Seguidores del vendedor", min_value=0, value=50
)

seller_pass_rate = st.sidebar.slider(
    "Tasa de aprobación del vendedor", 0.0, 1.0, 0.9
)

# CATEGÓRICAS
product_condition = st.sidebar.selectbox(
    "Condición del producto",
    ["Nuevo", "Muy bueno", "Bueno", "Usado"]
)

product_season = st.sidebar.selectbox(
    "Temporada",
    ["Primavera", "Verano", "Otoño", "Invierno"]
)

product_color = st.sidebar.selectbox(
    "Color",
    ["Negro", "Blanco", "Rojo", "Azul", "Beige", "Multicolor"]
)

usually_ships_within = st.sidebar.selectbox(
    "Tiempo de envío",
    ["1 día", "2 días", "3 días", "1 semana"]
)

has_cross_border_fees = st.sidebar.checkbox(
    "Tiene cargos internacionales"
)

# ================= ESTRUCTURA ORIGINAL (COLUMNAS) =================
col1, col2 = st.columns(2)

# ===== COLUMNA IMAGEN =====
with col1:
    st.subheader("📸 Imagen del Producto")

    opcion = st.radio(
        "Método de imagen",
        ["Subir imagen"]
    )

    if opcion == "Subir imagen":
        uploaded = st.file_uploader(
            "Selecciona una imagen",
            type=["jpg", "jpeg", "png"]
        )
    else:
        uploaded = st.camera_input("Tomar foto")

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Imagen ingresada", width=500)

# ===== COLUMNA RESULTADO =====
with col2:
    st.subheader("Resultado de Clasificación")

    if uploaded:
        x = preprocess_image(img)
        cat_out, attr_out = model.predict(x, verbose=0)

        # CATEGORÍA
        probs = cat_out[0]
        idx = int(np.argmax(probs))
        category = labels[idx] if labels and idx < len(labels) else str(idx)
        confidence = float(probs[idx])

        st.success("Clasificación completada")
        st.metric("Categoría", category)
        st.metric("Confianza", f"{confidence:.2f}")

# ================= ATRIBUTOS =================
if uploaded:
    attr_probs = attr_out[0]
    if not attr_names:
        attr_names = [f"attr_{i}" for i in range(len(attr_probs))]

    attr_df = pd.DataFrame({
        "Atributo": attr_names,
        "Probabilidad": attr_probs
    }).sort_values("Probabilidad", ascending=False)

    attr_df["Interpretación"] = [
        explain_attr(n, p) for n, p in zip(attr_df["Atributo"], attr_df["Probabilidad"])
    ]

    tab1, tab2, tab3 = st.tabs(["Usuario", "Detalles", "Atributos"])

    with tab1:
        st.subheader("🧾 Resumen")
        st.json({
            "likes": product_like_count,
            "seller_price": seller_price,
            "buyers_fees": buyers_fees,
            "seller_products_sold": seller_products_sold,
            "seller_num_followers": seller_num_followers,
            "seller_pass_rate": seller_pass_rate,
            "product_condition": product_condition,
            "product_season": product_season,
            "product_color": product_color,
            "usually_ships_within": usually_ships_within,
            "has_cross_border_fees": has_cross_border_fees
        })

    with tab2:
        st.subheader("🔬 Detalles técnicos")
        st.write(f"Índice categoría: {idx}")
        st.write(f"Confianza exacta: {confidence:.4f}")

    with tab3:
        st.subheader("🎯 Atributos detectados")
        st.dataframe(attr_df)

st.info("✅ App lista. Variables agregadas sin modificar la lógica existente.")
