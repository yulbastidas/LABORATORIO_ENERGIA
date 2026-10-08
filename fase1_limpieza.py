
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
ARCHIVO = BASE / "dataset_demanda_energia_LSTM_2025_2026.csv"
SALIDA = BASE / "resultados"
SALIDA.mkdir(exist_ok=True)

# 1. Cargar y explorar
df = pd.read_csv(ARCHIVO)

print("=== EXPLORACIÓN INICIAL ===")
print("Dimensiones:", df.shape)
print("\nTipos de datos:")
print(df.dtypes)
print("\nValores faltantes:")
print(df.isna().sum())
print("\nDuplicados exactos:", df.duplicated().sum())

df.isna().sum().rename("faltantes").to_csv(
    SALIDA / "reporte_faltantes.csv"
)

# 2. Convertir y validar fechas
df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce"
)

fechas_invalidas = df[df["timestamp"].isna()]
fechas_invalidas.to_csv(
    SALIDA / "fechas_invalidas.csv", index=False
)

df = df.dropna(subset=["timestamp"])

# 3. Eliminar duplicados exactos
duplicados = df[df.duplicated(keep="first")]
duplicados.to_csv(
    SALIDA / "duplicados_eliminados.csv", index=False
)

df = df.drop_duplicates().copy()

# 4. Normalizar texto sin modificar datos numéricos
df["periodo_dia"] = (
    df["periodo_dia"]
    .astype("string")
    .str.strip()
    .str.lower()
)

# 5. Orden cronológico
df = df.sort_values("timestamp").reset_index(drop=True)

# 6. Detectar fechas duplicadas conflictivas
conflictos = df[
    df.duplicated(subset=["timestamp"], keep=False)
]
conflictos.to_csv(
    SALIDA / "fechas_duplicadas_conflictivas.csv",
    index=False
)

# 7. Validar frecuencia horaria
horas_esperadas = pd.date_range(
    df["timestamp"].min(),
    df["timestamp"].max(),
    freq="h"
)
horas_faltantes = horas_esperadas.difference(
    pd.DatetimeIndex(df["timestamp"])
)
pd.DataFrame({
    "timestamp": horas_faltantes
}).to_csv(SALIDA / "horas_faltantes.csv", index=False)

# 8. Comprobar variables temporales
df["hora_calculada"] = df["timestamp"].dt.hour
df["dia_calculado"] = df["timestamp"].dt.dayofweek
df["mes_calculado"] = df["timestamp"].dt.month
df["fin_semana_calculado"] = (
    df["dia_calculado"] >= 5
).astype(int)

for original, calculada in [
    ("hora", "hora_calculada"),
    ("dia_semana", "dia_calculado"),
    ("mes", "mes_calculado"),
    ("fin_semana", "fin_semana_calculado")
]:
    cantidad = (df[original] != df[calculada]).sum()
    print(f"Inconsistencias {original}: {cantidad}")

# 9. Detectar anomalías físicas
reglas = {
    "demanda_mw": df["demanda_mw"] <= 0,
    "temperatura_c": (
        (df["temperatura_c"] < -50) |
        (df["temperatura_c"] > 60)
    ),
    "humedad_pct": (
        (df["humedad_pct"] < 0) |
        (df["humedad_pct"] > 100)
    ),
    "viento_kmh": df["viento_kmh"] < 0,
    "radiacion_wm2": df["radiacion_wm2"] < 0,
    "precipitacion_mm": df["precipitacion_mm"] < 0,
    "precio_kwh": df["precio_kwh"] < 0
}

anomalias = []
for variable, mascara in reglas.items():
    registros = df.loc[
        mascara.fillna(False), ["timestamp", variable]
    ].copy()
    registros["variable"] = variable
    registros["motivo"] = "Fuera de rango de revisión"
    registros = registros.rename(
        columns={variable: "valor"}
    )
    anomalias.append(registros)

reporte_anomalias = pd.concat(
    anomalias, ignore_index=True
)
reporte_anomalias.to_csv(
    SALIDA / "anomalias_detectadas.csv", index=False
)

# 10. Verificar coherencia del objetivo
# demanda_objetivo(t) debe ser demanda_mw(t + 1 hora)
siguiente = df[["timestamp", "demanda_mw"]].copy()
siguiente["timestamp"] -= pd.Timedelta(hours=1)
siguiente = siguiente.rename(
    columns={"demanda_mw": "demanda_siguiente_real"}
)

df = df.merge(
    siguiente, on="timestamp", how="left",
    validate="one_to_one"
)

comparables = (
    df["demanda_objetivo"].notna() &
    df["demanda_siguiente_real"].notna()
)
diferencias = (
    df["demanda_objetivo"] -
    df["demanda_siguiente_real"]
).abs()

print("\nObjetivos comparables:", comparables.sum())
print("Objetivos inconsistentes:",
      (comparables & (diferencias > 0.011)).sum())

df.loc[
    comparables & (diferencias > 0.011),
    ["timestamp", "demanda_objetivo",
     "demanda_siguiente_real"]
].to_csv(
    SALIDA / "objetivos_inconsistentes.csv",
    index=False
)

# 11. Guardar datos ordenados y auditados
columnas_auxiliares = [
    "hora_calculada", "dia_calculado",
    "mes_calculado", "fin_semana_calculado",
    "demanda_siguiente_real"
]

df.drop(columns=columnas_auxiliares).to_csv(
    SALIDA / "dataset_fase1_auditado.csv",
    index=False
)

# 12. Gráfica exploratoria
plt.figure(figsize=(13, 5))
plt.plot(
    df["timestamp"], df["demanda_mw"],
    linewidth=0.5
)
plt.title("Demanda energética histórica")
plt.xlabel("Fecha")
plt.ylabel("Demanda (MW)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    SALIDA / "demanda_historica.png", dpi=160
)
plt.close()

print("\n=== RESUMEN ===")
print("Registros auditados:", len(df))
print("Fechas conflictivas:", len(conflictos))
print("Horas faltantes:", len(horas_faltantes))
print("Anomalías físicas:", len(reporte_anomalias))
print("Archivos guardados en:", SALIDA)

print("\n=== ANOMALÍAS DETECTADAS ===")
print(reporte_anomalias.to_string(index=False))

print("\n=== OBJETIVOS INCONSISTENTES ===")
inconsistentes = df.loc[
    comparables & (diferencias > 0.011),
    ["timestamp", "demanda_objetivo",
     "demanda_siguiente_real"]
]
print(inconsistentes.to_string(index=False))


# 13. Preparación del dataset depurado

df_limpio = df.copy()

# Marcar valores físicamente inválidos
for variable, mascara in reglas.items():
    df_limpio.loc[mascara.fillna(False), variable] = np.nan

# Reconstruir objetivo de la siguiente hora
# La serie ya está ordenada y no presenta horas faltantes.
df_limpio["demanda_objetivo"] = (
    df_limpio["demanda_mw"].shift(-1)
)

# Retirar columnas auxiliares de auditoría
df_limpio = df_limpio.drop(
    columns=columnas_auxiliares
)

# El último registro no tiene objetivo observado
# y se conserva para posibles predicciones futuras.
df_limpio.to_csv(
    SALIDA / "dataset_limpio_fase1.csv",
    index=False
)

# Reporte final
print("\n=== LIMPIEZA FINAL ===")
print("Registros:", len(df_limpio))
print("Duplicados:", df_limpio.duplicated().sum())
print("Fechas duplicadas:",
      df_limpio["timestamp"].duplicated().sum())
print("\nValores faltantes finales:")
print(df_limpio.isna().sum())
print("\nPrimeras filas:")
print(df_limpio.head())
print("\nArchivo generado: dataset_limpio_fase1.csv")
