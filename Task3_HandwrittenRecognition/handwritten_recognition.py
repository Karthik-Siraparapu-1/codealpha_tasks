import numpy as np
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("Loading MNIST Dataset...")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining model...")

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

loss, accuracy = model.evaluate(X_test, y_test)

print("\nFinal Accuracy:", round(accuracy * 100, 2), "%")

predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, predicted_classes))

cm = confusion_matrix(y_test, predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training Accuracy")
plt.legend()
plt.savefig("outputs/training_accuracy.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training Loss")
plt.legend()
plt.savefig("outputs/training_loss.png")
plt.close()

sample_index = 15

sample_image = X_test[sample_index].reshape(1, 28, 28, 1)
sample_prediction = model.predict(sample_image)
sample_result = np.argmax(sample_prediction)

plt.figure(figsize=(6, 6))
plt.imshow(X_test[sample_index].reshape(28, 28), cmap="gray")
plt.title(f"Predicted Digit: {sample_result}")
plt.savefig("outputs/sample_prediction.png")
plt.close()

model.save("models/handwritten_model.keras")

report = {
    "accuracy": accuracy,
    "classification_report": classification_report(
        y_test,
        predicted_classes,
        output_dict=True
    )
}

joblib.dump(
    report,
    "models/model_report.pkl"
)

print("\nModel saved successfully!")
print("Reports generated successfully!")
print("Project completed.")