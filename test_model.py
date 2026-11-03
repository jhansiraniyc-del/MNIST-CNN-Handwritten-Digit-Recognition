import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Load MNIST test data

(_, _), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)


# Normalize

x_test = x_test.astype("float32") / 255.0


# Reshape

x_test = x_test.reshape(
    -1,
    28,
    28,
    1
)


# Load trained model

model = tf.keras.models.load_model(
    "mnist_cnn_model.keras"
)


print("Model loaded successfully!")


# Select an image

index = 100


image = x_test[index]


# Predict

prediction = model.predict(
    image.reshape(1, 28, 28, 1),
    verbose=0
)


predicted_digit = np.argmax(
    prediction
)


confidence = np.max(
    prediction
) * 100


actual_digit = y_test[index]


print(
    "Predicted digit:",
    predicted_digit
)

print(
    "Actual digit:",
    actual_digit
)

print(
    "Confidence:",
    round(confidence, 2),
    "%"
)


# Display image

plt.imshow(
    image.reshape(28, 28),
    cmap="gray"
)

plt.title(
    f"Predicted: {predicted_digit} | "
    f"Actual: {actual_digit}"
)

plt.axis("off")

plt.show()