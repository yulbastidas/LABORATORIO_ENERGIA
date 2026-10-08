
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

BASE = Path(__file__).resolve().parent
SALIDA = BASE / "resultados"
ARCHIVO = SALIDA / "dataset_limpio_fase1.csv"

df = pd.read_csv(ARCHIVO, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("=== FASE 2: PREPARACIÓN ===")
print("Registros:", len(df))

# 1. Validar la continuidad temporal
assert df["timestamp"].is_unique
assert df["timestamp"].diff().iloc[1:].eq(
    pd.Timedelta(hours=1)
).all(), "La serie presenta discontinuidades temporales"

# 2. Definir variables predictoras
variables = [
    "demanda_mw",
    "temperatura_c",
    "humedad_pct",
    "viento_kmh",
    "radiacion_wm2",
    "precipitacion_mm",
    "precio_kwh",
    "hora",
    "dia_semana",
    "fin_semana",
    "festivo",
    "mes"
]

# periodo_dia se representa mediante hora, evitando
# introducir una codificación ordinal artificial.

# 3. División cronológica 70 / 15 / 15
n = len(df)
fin_train = int(n * 0.70)
fin_val = int(n * 0.85)

print("\n=== DIVISIÓN CRONOLÓGICA ===")
print("Entrenamiento:", fin_train)
print("Validación:", fin_val - fin_train)
print("Prueba:", n - fin_val)

print("Fin entrenamiento:",
      df.loc[fin_train - 1, "timestamp"])
print("Inicio validación:",
      df.loc[fin_train, "timestamp"])
print("Inicio prueba:",
      df.loc[fin_val, "timestamp"])

# 4. Tratamiento causal de valores faltantes
# No se utilizan valores posteriores para rellenar
# observaciones anteriores.
datos = df[variables].copy()

# Valores de respaldo calculados SOLO con train.
medianas_train = datos.iloc[:fin_train].median()

# Forward fill conserva la última observación conocida.
datos = datos.ffill()

# Solo los faltantes iniciales sin dato previo
# se sustituyen por medianas de entrenamiento.
datos = datos.fillna(medianas_train)

assert not datos.isna().any().any()

# 5. Normalización
# Ajustar exclusivamente con entrenamiento.
scaler_x = StandardScaler()
scaler_y = StandardScaler()

scaler_x.fit(datos.iloc[:fin_train])

# El objetivo es la demanda real de la hora siguiente.
# Para ajustar el escalador se usan únicamente
# demandas observadas dentro del periodo train.
scaler_y.fit(
    df[["demanda_mw"]].iloc[:fin_train]
)

X_escalado = scaler_x.transform(datos)

# 6. Crear secuencias temporales
def crear_ventanas(tamano):
    X, y, indices = [], [], []

    # La ventana termina en t.
    # El objetivo corresponde a t + 1.
    for t in range(tamano - 1, len(df) - 1):
        inicio = t - tamano + 1

        ventana = X_escalado[inicio:t + 1]
        objetivo = df.loc[t + 1, "demanda_mw"]

        if pd.isna(objetivo):
            continue

        X.append(ventana)
        y.append(objetivo)
        indices.append(t + 1)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    indices = np.asarray(indices)

    y = scaler_y.transform(y).astype(np.float32)

    return X, y, indices


# 7. Generar las tres ventanas
for ventana in [12, 24, 48]:
    X, y, indices = crear_ventanas(ventana)

    train = indices < fin_train
    val = (indices >= fin_train) & (indices < fin_val)
    test = indices >= fin_val

    assert not np.any(train & val)
    assert not np.any(train & test)
    assert not np.any(val & test)

    np.savez_compressed(
        SALIDA / f"ventanas_{ventana}h.npz",
        X_train=X[train],
        y_train=y[train],
        X_val=X[val],
        y_val=y[val],
        X_test=X[test],
        y_test=y[test],
        indices_train=indices[train],
        indices_val=indices[val],
        indices_test=indices[test]
    )

    print(f"\n=== VENTANA {ventana} HORAS ===")
    print("Entrenamiento:", X[train].shape)
    print("Validación:", X[val].shape)
    print("Prueba:", X[test].shape)
    print("Objetivos entrenamiento:", y[train].shape)

# 8. Guardar normalizadores y metadatos
joblib.dump(scaler_x, SALIDA / "scaler_x.pkl")
joblib.dump(scaler_y, SALIDA / "scaler_y.pkl")

joblib.dump(
    {
        "variables": variables,
        "fin_train": fin_train,
        "fin_val": fin_val,
        "medianas_train": medianas_train.to_dict()
    },
    SALIDA / "configuracion_fase2.pkl"
)

print("\nFASE 2 COMPLETADA")
print("Secuencias y normalizadores guardados.")
