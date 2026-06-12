from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main() -> None:
    dataset_dir = Path(__file__).resolve().parent / "rockpaperscissors"
    if not dataset_dir.exists():
        raise FileNotFoundError(
            "Folder dataset tidak ditemukan. Letakkan hasil ekstrak dataset di: "
            + str(dataset_dir)
        )

    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        str(dataset_dir),
        target_size=(150, 150),
        batch_size=32,
        class_mode="categorical",
        subset="training",
    )

    validation_generator = train_datagen.flow_from_directory(
        str(dataset_dir),
        target_size=(150, 150),
        batch_size=32,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    class_indices = train_generator.class_indices
    index_to_class = {v: k for k, v in class_indices.items()}
    class_names = [index_to_class[i] for i in range(len(index_to_class))]
    print("Class indices:", class_indices)

    model = Sequential(
        [
            Input(shape=(150, 150, 3)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation="relu"),
            Dense(3, activation="softmax"),
        ]
    )

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=10,
    )

    validation_generator.reset()
    val_loss, val_acc = model.evaluate(validation_generator)
    print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")

    validation_generator.reset()
    predictions = model.predict(validation_generator)
    print("Predictions shape:", predictions.shape)
    print("Predictions (first 5 rows):")
    print(predictions[:5])

    predicted_classes = predictions.argmax(axis=1)
    true_classes = validation_generator.classes

    manual_acc = float(np.mean(predicted_classes == true_classes))
    print(f"Manual accuracy (from predict): {manual_acc}")

    cm = tf.math.confusion_matrix(
        true_classes, predicted_classes, num_classes=len(class_names)
    ).numpy()
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm_df)

    n_show = min(10, len(predicted_classes))
    print(f"Sample predictions (first {n_show} validation images):")
    for i in range(n_show):
        filename = validation_generator.filenames[i]
        true_label = class_names[int(true_classes[i])]
        pred_idx = int(predicted_classes[i])
        pred_label = class_names[pred_idx]
        conf = float(predictions[i, pred_idx])
        print(f"{i+1:03d} {filename} | true={true_label} pred={pred_label} conf={conf:.4f}")


if __name__ == "__main__":
    main()
