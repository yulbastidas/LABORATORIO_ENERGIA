
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from fase3_modelos_lstm import crear_modelo

BASE = Path(__file__).resolve().parent
SALIDA = BASE / "resultados"
MODELOS = SALIDA / "modelos"
MODELOS.mkdir(exist_ok=True)

SEMILLA = 42
random.seed(SEMILLA)
np.random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)

scaler_y = joblib.load(SALIDA / "scaler_y.pkl")

def evaluar(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)

    # Evitar división por cero en MAPE.
    validos = np.abs(y_real) > 1e-8
    mape = (
        np.mean(
            np.abs(
                (y_real[validos] - y_pred[validos])
                / y_real[validos]
            )
        ) * 100
        if np.any(validos) else np.nan
    )

    r2 = r2_score(y_real, y_pred)

    return mae, mse, rmse, mape, r2


resultados = []

for ventana in [12, 24, 48]:
    datos = np.load(
        SALIDA / f"ventanas_{ventana}h.npz"
    )

    X_train = datos["X_train"]
    y_train = datos["y_train"]
    X_val = datos["X_val"]
    y_val = datos["y_val"]
    X_test = datos["X_test"]
    y_test = datos["y_test"]

    for tipo in ["base", "profunda", "propuesta"]:
        nombre = f"{tipo}_{ventana}h"

        print(f"\n{'=' * 55}")
        print(f"ENTRENANDO: {nombre.upper()}")
        print(f"{'=' * 55}")

        tf.keras.backend.clear_session()
        tf.random.set_seed(SEMILLA)

        modelo = crear_modelo(
            tipo, ventana, X_train.shape[2]
        )

        ruta_modelo = MODELOS / f"{nombre}.keras"

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=str(ruta_modelo),
                monitor="val_loss",
                save_best_only=True
            )
        ]

        historial = modelo.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=60,
            batch_size=64,
            callbacks=callbacks,
            shuffle=False,
            verbose=1
        )

        pd.DataFrame(historial.history).to_csv(
            MODELOS / f"historial_{nombre}.csv",
            index=False
        )

        mejor_modelo = tf.keras.models.load_model(
            ruta_modelo
        )

        pred_scaled = mejor_modelo.predict(
            X_test, verbose=0
        )

        y_real = scaler_y.inverse_transform(
            y_test
        ).ravel()

        y_pred = scaler_y.inverse_transform(
            pred_scaled
        ).ravel()

        mae, mse, rmse, mape, r2 = evaluar(
            y_real, y_pred
        )

        pd.DataFrame({
            "indice": datos["indices_test"],
            "demanda_real_mw": y_real,
            "demanda_predicha_mw": y_pred,
            "error_mw": y_real - y_pred
        }).to_csv(
            MODELOS / f"predicciones_{nombre}.csv",
            index=False
        )

        resultados.append({
            "Modelo": tipo,
            "Ventana": ventana,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "R2": r2,
            "Epocas": len(historial.history["loss"]),
            "Mejor_epoca": (
                np.argmin(historial.history["val_loss"]) + 1
            ),
            "Loss_train_final": historial.history["loss"][-1],
            "Loss_val_final": historial.history["val_loss"][-1]
        })

        pd.DataFrame(resultados).to_csv(
            SALIDA / "comparacion_modelos.csv",
            index=False
        )

        print(f"\nRESULTADOS {nombre}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R²: {r2:.4f}")

print("\n=== ENTRENAMIENTO FINALIZADO ===")
print(pd.DataFrame(resultados).to_string(index=False))
