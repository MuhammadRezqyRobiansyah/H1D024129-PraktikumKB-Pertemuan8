from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


IRIS_LOCAL_CANDIDATES = [
    Path(__file__).resolve().parent / "iris" / "iris.data",
    Path(__file__).resolve().parent / "iris" / "bezdekIris.data",
]


def load_iris_dataset() -> tuple[np.ndarray, np.ndarray]:
    dataset: pd.DataFrame | None = None
    source: str | None = None

    for path in IRIS_LOCAL_CANDIDATES:
        if path.exists():
            dataset = pd.read_csv(path, header=None, sep=",")
            source = str(path)
            break

    if dataset is None:
        candidates = "\n".join(str(p) for p in IRIS_LOCAL_CANDIDATES)
        raise FileNotFoundError(
            "Dataset Iris lokal tidak ditemukan. Pastikan salah satu file berikut ada:\n"
            + candidates
        )

    print(f"Load dataset Iris dari: {source}")
    dataset = dataset.dropna(how="any")

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X = X.astype(np.float32)
    return X, y


def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(1000, activation="relu"),
            tf.keras.layers.Dense(500, activation="relu"),
            tf.keras.layers.Dense(300, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    return model


def plot_history(history: tf.keras.callbacks.History) -> None:
    import matplotlib.pyplot as plt

    pd.DataFrame(history.history).plot(figsize=(10, 6))
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def predict_new_data(model: tf.keras.Model, label_encoder: LabelEncoder) -> None:
    sepal_length = float(input("Masukkan sepal length: "))
    sepal_width = float(input("Masukkan sepal width: "))
    petal_length = float(input("Masukkan petal length: "))
    petal_width = float(input("Masukkan petal width: "))

    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]], dtype=np.float32)
    prediction = model.predict(new_data, verbose=0)

    predicted_class = prediction.argmax(axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    print(f"Prediksi kelas: {predicted_label[0]}")


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    X, y = load_iris_dataset()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = build_model(input_dim=X_train.shape[1])
    model.summary()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss: {loss}, Accuracy: {accuracy}")

    try:
        plot_history(history)
    except Exception as exc:
        print(f"Gagal plot history (matplotlib belum terpasang?): {exc}")

    predictions = model.predict(X_test, verbose=0)
    predicted_classes = predictions.argmax(axis=1)

    print("Prediksi:", predicted_classes)
    print("Label Asli:", y_test)

    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_test, predicted_classes)
        plot_confusion_matrix(cm, class_names=list(label_encoder.classes_))
    except Exception as exc:
        print(f"Gagal buat/plot confusion matrix: {exc}")

    try:
        answer = input("Coba prediksi data baru? (y/n): ").strip().lower()
    except EOFError:
        answer = "n"

    if answer in {"y", "yes"}:
        predict_new_data(model, label_encoder)


if __name__ == "__main__":
    main()
