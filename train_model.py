import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD MNIST DATASET
# ============================================================

print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

print("Dataset loaded successfully!")

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)

print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)


# ============================================================
# 2. DISPLAY A SAMPLE IMAGE
# ============================================================

plt.imshow(x_train[0], cmap="gray")

plt.title(
    "Sample Digit: " + str(y_train[0])
)

plt.axis("off")

plt.show()


# ============================================================
# 3. NORMALIZE PIXEL VALUES
# ============================================================

x_train = x_train.astype("float32") / 255.0

x_test = x_test.astype("float32") / 255.0


# ============================================================
# 4. RESHAPE IMAGES
# ============================================================

x_train = x_train.reshape(
    -1,
    28,
    28,
    1
)

x_test = x_test.reshape(
    -1,
    28,
    28,
    1
)


print("After reshaping:")
print("Training shape:", x_train.shape)
print("Testing shape:", x_test.shape)


# ============================================================
# 5. CREATE CUSTOM CNN
# ============================================================

model = tf.keras.Sequential([

    # First convolution layer
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    # First pooling layer
    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Second convolution layer
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    # Second pooling layer
    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Convert feature maps into one-dimensional vector
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Prevent overfitting
    tf.keras.layers.Dropout(0.5),

    # Output layer
    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# ============================================================
# 6. DISPLAY MODEL ARCHITECTURE
# ============================================================

print("\nCNN MODEL SUMMARY")
print("=================")

model.summary()


# ============================================================
# 7. COMPILE MODEL
# ============================================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\nStarting training...")

history = model.fit(

    x_train,

    y_train,

    epochs=10,

    batch_size=64,

    validation_split=0.1,

    verbose=1
)


# ============================================================
# 9. EVALUATE MODEL
# ============================================================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(

    x_test,

    y_test,

    verbose=1
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(
    "Test Loss:",
    test_loss
)

print(
    "Test Accuracy:",
    test_accuracy
)

print(
    "Test Accuracy (%):",
    test_accuracy * 100
)


# ============================================================
# 10. PLOT ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 11. PLOT LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 12. MAKE SAMPLE PREDICTIONS
# ============================================================

predictions = model.predict(
    x_test[:10]
)


plt.figure(figsize=(10, 5))


for i in range(10):

    predicted_digit = np.argmax(
        predictions[i]
    )

    actual_digit = y_test[i]

    plt.subplot(
        2,
        5,
        i + 1
    )

    plt.imshow(
        x_test[i].reshape(28, 28),
        cmap="gray"
    )

    plt.title(
        f"Pred: {predicted_digit}\n"
        f"Actual: {actual_digit}"
    )

    plt.axis("off")


plt.tight_layout()

plt.show()


# ============================================================
# 13. SAVE TRAINED MODEL
# ============================================================

model.save(
    "mnist_cnn_model.keras"
)

print("\nModel saved successfully!")

print(
    "File: mnist_cnn_model.keras"
)