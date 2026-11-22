"""Train and evaluate a CNN for MNIST handwritten digit recognition."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "mnist_cnn.keras"
OUTPUT_DIR = BASE_DIR / "outputs"


def build_model() -> tf.keras.Model:
    """Create a compact CNN for 28x28 grayscale digit images."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation="softmax"),
        ],
        name="mnist_cnn",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    tf.keras.utils.set_random_seed(42)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    model = build_model()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ]
    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4%}")

    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    print("\nClassification report:")
    print(classification_report(y_test, predictions, digits=4))

    model.save(MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Model accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Model loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close(figure)

    matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("MNIST confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
