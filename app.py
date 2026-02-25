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
METRICS_JSON = "data/processed/metrics.json"

# ================= STREAMLIT CONFIG =================
st.set_page_config(
    page_title="DeepFashion App",
    layout="wide"
)

st.title("DeepFashion - Clasificación de Prendas")
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
        st.error("No se encontró el modelo entrenado.")
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
            all_lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        lines = all_lines[2:] if len(all_lines) > 2 else []
        return [ln.split()[0] for ln in lines if ln]
    return None

@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_JSON):
        with open(METRICS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
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

def map_category_to_type(category):
    types = {
        "Camiseta": ["Jersey", "Tank", "Tee", "Henley"],
        "Chompa / Chaqueta": ["Anorak", "Bomber", "Cardigan", "Hoodie", "Jacket", "Parka", "Peacoat", "Sweater", "Coat", "Blazer", "Poncho", "Cape", "Turtleneck"],
        "Blusa / Top": ["Blouse", "Halter", "Top"],
        "Camisa": ["Button-Down", "Flannel"],
        "Pantalón": ["Capris", "Chinos", "Culottes", "Jeans", "Jeggings", "Jodhpurs", "Joggers", "Leggings", "Sweatpants", "Gauchos"],
        "Shorts / Pantalón Corto": ["Cutoffs", "Shorts", "Sweatshorts", "Trunks"],
        "Vestido / Enterizo": ["Caftan", "Dress", "Kaftan", "Nightdress", "Jumpsuit", "Kimono", "Coverup"],
        "Falda": ["Sarong", "Skirt"]
    }
    
    cat = category.title()
    for type_name, labels in types.items():
        if cat in labels:
            return type_name
    return "Otro / No clasificado"

def translate_exact_category(category):
    translations = {
        "Anorak": "Anorak", "Blazer": "Blazer", "Blouse": "Blusa", "Bomber": "Chaqueta Bomber",
        "Button-Down": "Camisa de botones", "Cardigan": "Cárdigan", "Flannel": "Camisa de franela",
        "Halter": "Top Halter", "Henley": "Camiseta Henley", "Hoodie": "Sudadera (Hoodie)",
        "Jacket": "Chaqueta", "Jersey": "Jersey / Suéter", "Parka": "Parka", "Peacoat": "Chaquetón",
        "Poncho": "Poncho", "Sweater": "Suéter", "Tank": "Camiseta de tirantes", "Tee": "Camiseta (T-shirt)",
        "Top": "Top", "Turtleneck": "Cuello de tortuga", "Capris": "Pantalón Capri",
        "Chinos": "Pantalón Chino", "Culottes": "Pantalón Culotte", "Cutoffs": "Shorts recortados",
        "Gauchos": "Pantalón Gaucho", "Jeans": "Jeans / Vaqueros", "Jeggings": "Jeggings",
        "Jodhpurs": "Pantalón Jodhpurs", "Joggers": "Joggers (Deportivo)", "Leggings": "Leggings",
        "Sarong": "Pareo", "Shorts": "Shorts", "Skirt": "Falda", "Sweatpants": "Pantalón de chándal",
        "Sweatshorts": "Shorts de chándal", "Trunks": "Bañador", "Caftan": "Caftán",
        "Cape": "Capa", "Coat": "Abrigo", "Coverup": "Salida de baño", "Dress": "Vestido",
        "Jumpsuit": "Enterizo / Jumpsuit", "Kaftan": "Kaftán", "Kimono": "Kimono", "Nightdress": "Camisón"
    }
    return translations.get(category.title(), category)

def get_dominant_color(img: Image.Image):
    img_small = img.convert("RGB").resize((60, 60))
    w, h = img_small.size
    center = img_small.crop((w//5, h//5, 4*w//5, 4*h//5))
    pixels = np.array(center).reshape(-1, 3).astype(float)
    n_clusters = 4 
    centers = pixels[np.random.choice(pixels.shape[0], n_clusters, replace=False)]
    
    for _ in range(10): 
        distances = np.linalg.norm(pixels[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = np.array([pixels[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] 
                               for i in range(n_clusters)])
        if np.allclose(centers, new_centers, atol=1.0):
            break
        centers = new_centers

    unique, counts = np.unique(labels, return_counts=True)
    sorted_indices = unique[np.argsort(counts)[::-1]]
    
    dominant_idx = sorted_indices[0]
    for idx in sorted_indices:
        color = centers[idx]
        if not (np.all(color > 245) or np.all(color < 10)):
            dominant_idx = idx
            break
            
    avg_color = centers[dominant_idx].astype(int)
    return avg_color

def map_color_to_name(rgb):
    colors = {
        "Negro": [20, 20, 20],
        "Blanco": [240, 240, 240],
        "Rojo": [200, 30, 30],
        "Verde": [30, 150, 30],
        "Azul": [30, 30, 200],
        "Gris": [128, 128, 128],
        "Beige": [245, 245, 220],
        "Marrón": [100, 50, 20],
        "Amarillo": [255, 255, 50],
        "Naranja": [255, 120, 20],
        "Violeta": [180, 50, 180],
        "Celeste": [170, 220, 255],
        "Rosa": [255, 180, 190],
        "Jean/Denim": [50, 80, 120],
        "Vino/Burdeos": [128, 0, 32],
        "Verde Olivo": [85, 107, 47],
        "Azul Marino": [0, 0, 128],
        "Gris Claro": [211, 211, 211]
    }
    min_dist = float('inf')
    closest_name = "Desconocido"
    for name, c_rgb in colors.items():
        dist = np.linalg.norm(np.array(rgb) - np.array(c_rgb))
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

# ================= CARGA MODELO =================
model = load_model()
labels = load_labels()
attr_names = load_attributes()
metrics = load_metrics()

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
    st.subheader("Imagen del Producto")

    opcion = st.radio(
        "Método de imagen",
        ["Subir imagen", "Tomar fotografía"]
    )

    if opcion == "Subir imagen":
        uploaded = st.file_uploader(
            "Selecciona una imagen",
            type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "jfif"]
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

        # COLOR
        dom_color = get_dominant_color(img)
        color_name = map_color_to_name(dom_color)
        
        # TIPO DE PRENDA
        cloth_type = map_category_to_type(category)
        exact_type = translate_exact_category(category)

        st.success("Clasificación completada")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Tipo de Ropa", cloth_type)
            st.metric("Prenda Exacta", exact_type)
        with m_col2:
            st.metric("Color Detectado", color_name)
            st.metric("Confianza", f"{confidence:.2f}")
            # Mostrar parche de color
            st.markdown(
                f'<div style="background-color: rgb({dom_color[0]},{dom_color[1]},{dom_color[2]}); '
                f'width: 50px; height: 50px; border-radius: 5px; border: 1px solid #ddd;"></div>',
                unsafe_allow_html=True
            )
        
        # PRECISIÓN ADICIONAL
        if metrics:
            st.markdown("---")
            st.subheader("Métricas de Rendimiento del Modelo")
            p_col1, p_col2 = st.columns(2)
            
            # Precisión general
            overall_p = metrics.get("overall_precision", 0) * 100
            p_col1.metric("Precisión General", f"{overall_p:.1f}%")
            
            # Precisión específica por categoría
            cat_precisions = metrics.get("per_category_precision", {})
            cat_p = cat_precisions.get(category.title(), metrics.get("overall_accuracy", 0.9)) * 100
            p_col2.metric(f"Precisión ({exact_type})", f"{cat_p:.1f}%")
            
            st.caption("Nota: La 'Confianza' es para esta imagen específica, mientras que la 'Precisión' es el rendimiento total del modelo para este tipo de prenda.")

# ================= ATRIBUTOS =================
if uploaded:
    attr_probs = attr_out[0]
    num_pred_attrs = len(attr_probs)
    
    if not attr_names:
        attr_names = [f"attr_{i}" for i in range(num_pred_attrs)]
    elif len(attr_names) < num_pred_attrs:
        padding = [f"attr_{i}" for i in range(len(attr_names), num_pred_attrs)]
        attr_names = attr_names + padding
    elif len(attr_names) > num_pred_attrs:
        attr_names = attr_names[:num_pred_attrs]

    attr_df = pd.DataFrame({
        "Atributo": attr_names,
        "Probabilidad": attr_probs
    }).sort_values("Probabilidad", ascending=False)

    attr_df["Interpretación"] = [
        explain_attr(n, p) for n, p in zip(attr_df["Atributo"], attr_df["Probabilidad"])
    ]

    tab1, tab2, tab3, tab4 = st.tabs(["Resumen de Características", "Variables", "Detalles", "Atributos Completos"])

    with tab1:
        st.subheader("Características Detectadas")
        attr_filtered = attr_df[attr_df["Probabilidad"] > 0.25].head(100)
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.write("**Tipo de Tela / Textura:**")
            fabric_keywords = ["cotton", "denim", "wool", "silk", "leather", "lace", "chiffon", "knit"]
            fabrics = [n for n in attr_filtered["Atributo"].head(20) if any(k in n.lower() for k in fabric_keywords)]
            if fabrics:
                count = 0
                for f in fabrics:
                    if count >= 3: break
                    st.info(f"{f}")
                    count += 1
            else:
                top_attr = attr_filtered.iloc[0]["Atributo"] if not attr_filtered.empty else "Estándar"
                st.info(f"Probable: {top_attr}")

            st.write("**Bolsillos y Detalles:**")
            pocket_keywords = ["pocket", "button", "zipper", "collar", "hooded", "cargo", "utility"]
            parts = [n for n in attr_filtered["Atributo"] if any(k in n.lower() for k in pocket_keywords)]
            count_pockets = [n for n in parts if "5-pocket" in n.lower()]
            if count_pockets:
                st.success(f"Detección precisa: {count_pockets[0]}")
            if parts:
                count = 0
                for p in parts:
                    if count >= 3: break
                    st.info(f"{p}")
                    count += 1
            else:
                st.write("No se detectaron detalles adicionales.")

        with col_c2:
            st.write("**Tamaño / Forma:**")
            # Mapeo español para términos técnicos de forma y tamaño
            shape_map = {
                "short": "Corto",
                "long": "Largo",
                "mini": "Mini (Muy corto)",
                "midi": "Largo medio (Midi)",
                "maxi": "Maxi (Largo total)",
                "sleeveless": "Sin mangas",
                "oversized": "Holgado / Oversized",
                "cropped": "Corto / Recortado",
                "slim": "Ajustado / Slim",
                "loose": "Suelto / Holgado",
                "fit": "Ajustado",
                "regular": "Ajuste estándar"
            }
         
            detected_shapes = []
            for attr in attr_filtered["Atributo"].tolist():
                for key, spanish in shape_map.items():
                    if key in attr.lower():
                        detected_shapes.append(spanish)
                        break
            
            detected_shapes = list(dict.fromkeys(detected_shapes))
            
            if detected_shapes:
                for s in detected_shapes[:3]:
                    st.info(s)
            else:
                st.write("No se detectaron formas específicas distintivas.")

            st.write("**Color Dominante:**")
            st.success(f"{color_name}")

    with tab2:
        st.subheader("Variables de Usuario")
        st.json({
            "likes": product_like_count,
            "seller_price": seller_price,
            "buyers_fees": buyers_fees,
            "seller_products_sold": seller_products_sold,
            "seller_num_followers": seller_num_followers,
            "seller_pass_rate": seller_pass_rate,
            "product_condition": product_condition,
            "product_season": product_season,
            "product_color": color_name,
            "product_exact_type": exact_type,
            "product_category": category,
            "product_type": cloth_type,
            "usually_ships_within": usually_ships_within,
            "has_cross_border_fees": has_cross_border_fees
        })

    with tab3:
        st.subheader("Detalles técnicos")
        st.write(f"Índice categoría: {idx}")
        st.write(f"Confianza exacta: {confidence:.4f}")
        st.write(f"RGB dominante: {dom_color}")

    with tab4:
        st.subheader("Todos los Atributos")
        st.dataframe(attr_df)

st.info("Color, Tela, Tamaño y Bolsillos.")