
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2

def crear_modelo(tipo, ventana, num_variables=12):
    tf.keras.backend.clear_session()

    modelo = Sequential(name=f"lstm_{tipo}_{ventana}h")
    modelo.add(Input(shape=(ventana, num_variables)))

    if tipo == "base":
        modelo.add(LSTM(32))
        modelo.add(Dense(1))

    elif tipo == "profunda":
        modelo.add(LSTM(64, return_sequences=True))
        modelo.add(LSTM(32))
        modelo.add(Dense(1))

    elif tipo == "propuesta":
        modelo.add(
            LSTM(
                64,
                return_sequences=True,
                kernel_regularizer=l2(0.0001)
            )
        )
        modelo.add(Dropout(0.2))
        modelo.add(LSTM(32))
        modelo.add(Dropout(0.2))
        modelo.add(Dense(16, activation="relu"))
        modelo.add(Dense(1))

    else:
        raise ValueError("Arquitectura no reconocida")

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    return modelo


if __name__ == "__main__":
    for tipo in ["base", "profunda", "propuesta"]:
        modelo = crear_modelo(tipo, ventana=24)

        print(f"\n=== MODELO {tipo.upper()} ===")
        modelo.summary()
