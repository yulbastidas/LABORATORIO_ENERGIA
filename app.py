from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Energy AI · Predicción LSTM", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "resultados"
MODELS = RESULTS / "modelos"
FEATURES = ["demanda_mw", "temperatura_c", "humedad_pct", "viento_kmh", "radiacion_wm2", "precipitacion_mm", "precio_kwh", "hora", "dia_semana", "fin_semana", "festivo", "mes"]
LABELS = {"demanda_mw": "Demanda (MW)", "temperatura_c": "Temperatura (°C)", "humedad_pct": "Humedad (%)", "viento_kmh": "Viento (km/h)", "radiacion_wm2": "Radiación (W/m²)", "precipitacion_mm": "Precipitación (mm)", "precio_kwh": "Precio (kWh)", "hora": "Hora (0–23)", "dia_semana": "Día semana (0–6)", "fin_semana": "Fin de semana", "festivo": "Festivo", "mes": "Mes (1–12)"}
PALETTE = {"base": "#4ea8ff", "profunda": "#a78bfa", "propuesta": "#2dd4bf"}

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 83% 1%,#152d3c 0%,transparent 36%),#090f1d;color:#ecf5ff}
[data-testid="stSidebar"]{background:#101a2b;border-right:1px solid #24364b}
.block-container{padding-top:1.6rem;max-width:1440px}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em}
[data-testid="stMetric"]{background:linear-gradient(145deg,#16273c,#101b2e);padding:20px;border:1px solid #2a4057;border-radius:17px}
[data-testid="stMetricLabel"]{color:#a9bdd0}
[data-testid="stMetricValue"]{font-family:'Space Grotesk',sans-serif;font-size:1.85rem;color:#e9f7ff}
.stTabs [data-baseweb="tab-list"]{gap:12px;border-bottom:1px solid #283c52}
.stTabs [data-baseweb="tab"]{background:#121e31;border-radius:11px 11px 0 0;padding:11px 19px}
.stTabs [aria-selected="true"]{background:#193448!important;color:#5eead4!important}
.stButton>button[kind="primary"]{background:linear-gradient(100deg,#06b6d4,#14b8a6);border:0;color:#04121c;font-weight:800;border-radius:12px;min-height:48px}
.stButton>button[kind="primary"]:hover{background:linear-gradient(100deg,#22d3ee,#5eead4);color:#04121c}
[data-testid="stFileUploader"],div[data-testid="stExpander"]{border:1px solid #2a4057;border-radius:14px}
.hero{border:1px solid #254e60;border-radius:22px;padding:28px 30px;background:linear-gradient(115deg,#122b3d,#112039 68%,#17253e);margin-bottom:22px}
.hero h1{margin:0;font-size:2.6rem;color:#f1fbff}
.hero p{color:#a6c5d6;margin:7px 0 0}
.eyebrow{font-size:.72rem;font-weight:800;letter-spacing:.18em;color:#5eead4}
.note{background:#11273b;border-left:3px solid #2dd4bf;border-radius:9px;padding:13px 16px;color:#b8d8e6}
.small{color:#a9bdd0;font-size:.88rem}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    path = RESULTS / "dataset_limpio_fase1.csv"
    data = pd.read_csv(path)
    missing = [x for x in FEATURES if x not in data.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {missing}")
    for name in FEATURES:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        data = data.sort_values("timestamp").reset_index(drop=True)
    return data

@st.cache_resource(show_spinner=False)
def load_scalers():
    return joblib.load(RESULTS / "scaler_x.pkl"), joblib.load(RESULTS / "scaler_y.pkl")

@st.cache_resource(show_spinner="Cargando red neuronal LSTM…")
def load_model(model, window):
    return tf.keras.models.load_model(MODELS / f"{model}_{window}h.keras", compile=False)

@st.cache_data(show_spinner=False)
def load_comparison():
    path = RESULTS / "comparacion_modelos.csv"
    if not path.exists():
        return pd.DataFrame()
    result = pd.read_csv(path)
    result.columns = result.columns.str.strip().str.lower()
    if "modelo" in result:
        result["modelo"] = result["modelo"].astype(str).str.strip().str.lower()
    return result

@st.cache_data(show_spinner=False)
def training_medians(data):
    # En fase 2, entrenamiento = 70% inicial. Medianas calculadas únicamente allí.
    n_train = int(len(data) * 0.70)
    return data.iloc[:n_train][FEATURES].median().fillna(0)

def validate_features(frame):
    problems = []
    for col, lo, hi in [("humedad_pct",0,100),("viento_kmh",0,None),("radiacion_wm2",0,None),("precipitacion_mm",0,None),("precio_kwh",0,None),("hora",0,23),("dia_semana",0,6),("mes",1,12)]:
        vals = frame[col].dropna()
        if (lo is not None and (vals < lo).any()) or (hi is not None and (vals > hi).any()):
            problems.append(LABELS[col])
    for col in ["fin_semana", "festivo"]:
        if not frame[col].dropna().isin([0,1]).all():
            problems.append(LABELS[col])
    for col in ["hora", "dia_semana", "mes"]:
        if not np.all(np.isclose(frame[col].dropna() % 1, 0)):
            problems.append(LABELS[col])
    return problems

def prepare_sequence(source, window, medians):
    if len(source) < window:
        raise ValueError(f"Se requieren al menos {window} filas horarias; recibidas: {len(source)}.")
    seq = source[FEATURES].tail(window).copy()
    seq = seq.ffill().fillna(medians)
    if seq.isna().any().any():
        raise ValueError("Quedan valores faltantes sin resolver en la secuencia.")
    problems = validate_features(seq)
    if problems:
        raise ValueError("Valores fuera de rango: " + ", ".join(problems))
    return seq

def predict(seq, model_name, window, sx, sy):
    matrix = sx.transform(seq[FEATURES]).astype(np.float32)
    matrix = matrix.reshape(1, window, len(FEATURES))
    prediction_scaled = load_model(model_name, window).predict(matrix, verbose=0)
    return float(sy.inverse_transform(np.asarray(prediction_scaled).reshape(-1,1))[0,0])

def line_layout(fig, title=None, height=370):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,23,39,.55)", font=dict(color="#bed3e5"), margin=dict(l=12,r=12,t=55,b=25), height=height, title=title, legend=dict(orientation="h",y=1.12,x=0), hovermode="x unified")
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(151,179,205,.12)",zeroline=False)
    return fig

try:
    df = load_data()
    scaler_x, scaler_y = load_scalers()
    comparison = load_comparison()
    medians = training_medians(df)
except Exception as exc:
    st.error(f"No se pudieron cargar los datos del proyecto: {exc}")
    st.info("Guarda este archivo como app.py en LABORATORIO y conserva la carpeta resultados junto a él.")
    st.stop()

with st.sidebar:
    st.markdown("### ⚡ ENERGY AI")
    st.caption("LABORATORIO DE INTELIGENCIA ARTIFICIAL")
    st.divider()
    model_name = st.selectbox("Arquitectura LSTM", ["propuesta", "base", "profunda"], format_func=str.capitalize)
    window = st.selectbox("Ventana temporal", [48,12,24], format_func=lambda x:f"{x} horas")
    model_path = MODELS / f"{model_name}_{window}h.keras"
    if model_path.exists():
        st.success("Modelo entrenado disponible")
    else:
        st.error(f"No existe {model_path.name}")
    st.divider()
    st.caption("12 variables predictoras · 3 arquitecturas · 3 ventanas")
    st.caption("Predicción a un paso: siguiente hora")

st.markdown('<div class="hero"><div class="eyebrow">PLATAFORMA DE ANÁLISIS PREDICTIVO</div><h1>⚡ Energy AI</h1><p>Predicción inteligente de demanda eléctrica · Redes neuronales LSTM · Horizonte de una hora</p></div>', unsafe_allow_html=True)

best = comparison.sort_values("mae").iloc[0] if not comparison.empty and "mae" in comparison else None
last_demand = float(df["demanda_mw"].dropna().iloc[-1])
c1,c2,c3,c4 = st.columns(4)
c1.metric("Demanda del último registro", f"{last_demand:,.2f} MW")
c2.metric("Modelos entrenados", "9 / 9")
c3.metric("Mejor MAE en pruebas", f"{best['mae']:.2f} MW" if best is not None else "—")
c4.metric("Mejor MAPE en pruebas", f"{comparison['mape'].min():.2f}%" if not comparison.empty and "mape" in comparison else "—")

inicio, simulador, modelos_tab, analisis = st.tabs(["🏠 Resumen", "⚡ Simulador", "📊 Modelos", "📈 Análisis"])

with inicio:
    st.subheader("Panorama de demanda eléctrica")
    n_hours = st.select_slider("Periodo histórico", options=[24,48,72,168,336,720], value=168, format_func=lambda n:f"{n} horas")
    historic = df.tail(n_hours).copy()
    x_axis = "timestamp" if "timestamp" in historic and historic["timestamp"].notna().all() else None
    if x_axis is None:
        historic["registro"] = range(len(historic))
        x_axis = "registro"
    fig = px.area(historic,x=x_axis,y="demanda_mw",labels={"demanda_mw":"Demanda (MW)",x_axis:"Fecha y hora"},color_discrete_sequence=["#2dd4bf"])
    fig.update_traces(line_width=2,fillcolor="rgba(45,212,191,.12)")
    st.plotly_chart(line_layout(fig,"Comportamiento reciente de la demanda",430),use_container_width=True)
    p1,p2,p3 = st.columns(3)
    p1.metric("Demanda media del periodo",f"{historic['demanda_mw'].mean():.2f} MW")
    p2.metric("Demanda máxima",f"{historic['demanda_mw'].max():.2f} MW")
    p3.metric("Demanda mínima",f"{historic['demanda_mw'].min():.2f} MW")
    st.markdown('<div class="note">El histórico muestra observaciones del dataset. Una predicción nueva se calcula en la pestaña <b>Simulador</b>, usando un modelo LSTM entrenado.</div>',unsafe_allow_html=True)

with simulador:
    st.subheader("Simulador de la próxima hora")
    st.write(f"Modelo seleccionado: **{model_name.capitalize()} · {window} horas**. Cada predicción utiliza {window} observaciones consecutivas de las 12 variables.")
    source_type = st.radio("Origen de los datos",["Histórico del proyecto", "Subir CSV independiente"],horizontal=True)
    source = df.copy()
    if source_type == "Subir CSV independiente":
        st.caption("El CSV debe tener al menos tantas filas como la ventana seleccionada y las 12 columnas predictoras. No incluyas la demanda futura como entrada.")
        template = df[FEATURES].tail(window).to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Descargar plantilla CSV",template,file_name=f"plantilla_{window}h.csv",mime="text/csv")
        upload = st.file_uploader("Selecciona el CSV horario",type=["csv"])
        if upload is None:
            st.info("Carga tu CSV para habilitar las predicciones independientes.")
            st.stop()
        try:
            source = pd.read_csv(upload)
            missing = [col for col in FEATURES if col not in source.columns]
            if missing:
                raise ValueError("Faltan columnas: " + ", ".join(missing))
            for col in FEATURES:
                source[col] = pd.to_numeric(source[col],errors="coerce")
            if "timestamp" in source.columns:
                source["timestamp"] = pd.to_datetime(source["timestamp"],errors="raise")
                source = source.sort_values("timestamp")
                if source["timestamp"].duplicated().any():
                    raise ValueError("El CSV tiene horas duplicadas.")
                last_hours = source["timestamp"].tail(window)
                if len(last_hours) >= 2 and not last_hours.diff().dropna().eq(pd.Timedelta(hours=1)).all():
                    raise ValueError("Las últimas horas del CSV no son consecutivas.")
        except Exception as exc:
            st.error(f"CSV no válido: {exc}")
            st.stop()
    try:
        sequence = prepare_sequence(source,window,medians)
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    st.markdown("#### Escenario de predicción")
    st.caption("Los controles modifican únicamente la última hora de la secuencia. Las horas anteriores se mantienen como contexto histórico.")
    with st.form("prediction_form"):
        with st.expander("✏️ Ajustar variables de la última hora",expanded=True):
            inputs = {}
            cols = st.columns(3)
            for idx, feature in enumerate(FEATURES):
                with cols[idx%3]:
                    value = float(sequence.iloc[-1][feature])
                    if feature in ["fin_semana","festivo"]:
                        inputs[feature] = st.selectbox(LABELS[feature],[0,1],index=1 if value >= .5 else 0)
                    elif feature in ["hora","dia_semana","mes"]:
                        limits = {"hora":(0,23),"dia_semana":(0,6),"mes":(1,12)}
                        lo,hi=limits[feature]
                        inputs[feature] = st.number_input(LABELS[feature],min_value=lo,max_value=hi,value=int(np.clip(round(value),lo,hi)),step=1)
                    else:
                        bounds = {"humedad_pct":(0.,100.),"viento_kmh":(0.,None),"radiacion_wm2":(0.,None),"precipitacion_mm":(0.,None),"precio_kwh":(0.,None)}
                        lo,hi = bounds.get(feature,(None,None))
                        value = max(lo,value) if lo is not None else value
                        value = min(hi,value) if hi is not None else value
                        inputs[feature] = st.number_input(LABELS[feature],min_value=lo,max_value=hi,value=float(value),step=.1,format="%.2f")
        submitted = st.form_submit_button("⚡ CALCULAR PREDICCIÓN LSTM",type="primary",use_container_width=True)
    with st.expander(f"🔍 Inspeccionar las {window} horas de entrada"):
        st.dataframe(sequence,use_container_width=True)
    if submitted:
        try:
            edited = sequence.copy()
            edited.loc[edited.index[-1],FEATURES] = [inputs[name] for name in FEATURES]
            issues = validate_features(edited)
            if issues:
                raise ValueError("Valores no válidos: " + ", ".join(issues))
            result = predict(edited,model_name,window,scaler_x,scaler_y)
            current = float(edited.iloc[-1]["demanda_mw"])
            delta = result-current
            st.success("Predicción calculada con el modelo neuronal guardado.")
            r1,r2,r3 = st.columns(3)
            r1.metric("Demanda de la última hora",f"{current:.2f} MW")
            r2.metric("Demanda prevista (+1h)",f"{result:.2f} MW",delta=f"{delta:+.2f} MW")
            r3.metric("Variación prevista",f"{delta/current*100:+.2f}%" if current else "No calculable")
            chart = go.Figure()
            chart.add_trace(go.Scatter(x=list(range(-window+1,1)),y=edited["demanda_mw"],name="Histórico",mode="lines",line=dict(color="#4ea8ff",width=3)))
            chart.add_trace(go.Scatter(x=[0,1],y=[current,result],name="Pronóstico +1h",mode="lines+markers",line=dict(color="#2dd4bf",width=3,dash="dash"),marker=dict(size=11)))
            chart.update_layout(xaxis_title="Horas relativas a la última observación",yaxis_title="Demanda (MW)")
            st.plotly_chart(line_layout(chart,"Demanda histórica y pronóstico",420),use_container_width=True)
            st.caption("Esta es una estimación del modelo, no una medición futura ni una garantía de precisión para el escenario introducido.")
        except Exception as exc:
            st.error(f"No fue posible predecir: {exc}")

with modelos_tab:
    st.subheader("Comparación experimental · 9 configuraciones")
    if comparison.empty:
        st.warning("No se encontró comparacion_modelos.csv")
    else:
        show = comparison.copy()
        st.dataframe(show.style.format({c:"{:.3f}" for c in ["mae","mse","rmse","mape","r2"] if c in show.columns}),use_container_width=True,hide_index=True)
        metric = st.selectbox("Métrica",[c for c in ["mae","rmse","mape","r2","mse"] if c in show.columns],format_func=str.upper)
        show["Configuración"] = show["modelo"].str.capitalize()+" · "+show["ventana"].astype(int).astype(str)+"h"
        fig = px.bar(show,x="Configuración",y=metric,color="modelo",color_discrete_map=PALETTE,text_auto=".2f")
        fig.update_layout(yaxis_title=metric.upper(),xaxis_title="")
        st.plotly_chart(line_layout(fig,f"Resultados de {metric.upper()} por modelo",430),use_container_width=True)
        if "mae" in show:
            winner = show.sort_values("mae").iloc[0]
            st.info(f"Menor MAE en prueba: {winner['modelo'].capitalize()} con {int(winner['ventana'])}h ({winner['mae']:.2f} MW). Esta comparación describe pruebas; para seleccionar hiperparámetros debe utilizarse validación.")

with analisis:
    st.subheader("Diagnóstico de aprendizaje y errores")
    ca,cb = st.columns(2)
    with ca:
        analysis_model = st.selectbox("Arquitectura",["base","profunda","propuesta"],index=2,key="analysis_model",format_func=str.capitalize)
    with cb:
        analysis_window = st.selectbox("Ventana",[12,24,48],index=2,key="analysis_window",format_func=lambda n:f"{n} horas")
    hpath = MODELS / f"historial_{analysis_model}_{analysis_window}h.csv"
    ppath = MODELS / f"predicciones_{analysis_model}_{analysis_window}h.csv"
    if hpath.exists():
        history = pd.read_csv(hpath)
        history.columns = history.columns.str.strip().str.lower()
        if {"loss","val_loss"}.issubset(history.columns):
            history["Época"] = np.arange(1,len(history)+1)
            hfig = go.Figure()
            hfig.add_trace(go.Scatter(x=history["Época"],y=history["loss"],name="Entrenamiento",line=dict(color="#4ea8ff")))
            hfig.add_trace(go.Scatter(x=history["Época"],y=history["val_loss"],name="Validación",line=dict(color="#2dd4bf")))
            hfig.update_layout(xaxis_title="Época",yaxis_title="Pérdida (MSE escalado)")
            st.plotly_chart(line_layout(hfig,"Curvas de aprendizaje",390),use_container_width=True)
            st.caption(f"Mejor época según val_loss: {int(history['val_loss'].idxmin()+1)} de {len(history)}.")
        else:
            st.dataframe(history,use_container_width=True)
    else:
        st.warning(f"No se encontró {hpath.name}")
    if ppath.exists():
        pred = pd.read_csv(ppath)
        pred.columns = pred.columns.str.strip().str.lower()
        real_candidates = [x for x in pred if "real" in x or "true" in x]
        predicted_candidates = [x for x in pred if "pred" in x]
        if real_candidates and predicted_candidates:
            real, estimated = real_candidates[0], predicted_candidates[0]
            valid = pred[[real,estimated]].apply(pd.to_numeric,errors="coerce").dropna()
            if not valid.empty:
                n = st.slider("Observaciones de prueba a visualizar",min_value=50,max_value=min(500,len(valid)),value=min(200,len(valid)),step=10) if len(valid)>=50 else len(valid)
                view = valid.head(n)
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=view[real],name="Real",line=dict(color="#4ea8ff")))
                fig.add_trace(go.Scatter(y=view[estimated],name="Predicción",line=dict(color="#2dd4bf")))
                fig.update_layout(xaxis_title="Observación de prueba",yaxis_title="Demanda (MW)")
                st.plotly_chart(line_layout(fig,"Demanda real frente a predicha",390),use_container_width=True)
                errors = valid[real]-valid[estimated]
                ef = px.histogram(x=errors,nbins=45,color_discrete_sequence=["#a78bfa"],labels={"x":"Error (MW)","y":"Frecuencia"})
                st.plotly_chart(line_layout(ef,"Distribución de errores",330),use_container_width=True)
                st.metric("Error medio absoluto del archivo",f"{errors.abs().mean():.2f} MW")
        else:
            st.warning("Columnas disponibles en predicciones: " + ", ".join(pred.columns) + ". Revisa los nombres de las columnas real y predicha.")
            st.dataframe(pred.head(10),use_container_width=True)
    else:
        st.warning(f"No se encontró {ppath.name}")

st.divider()
st.caption("ENERGY AI · Proyecto académico de Inteligencia Artificial · Predicción horaria LSTM · Datos y modelos locales")
