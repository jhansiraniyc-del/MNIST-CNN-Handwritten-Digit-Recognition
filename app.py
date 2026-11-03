from flask import Flask, render_template, request, jsonify

import tensorflow as tf
import numpy as np

from PIL import Image, ImageOps

import base64
import io


# ============================================================
# CREATE FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD TRAINED CNN MODEL
# ============================================================

model = tf.keras.models.load_model(
    "mnist_cnn_model.keras"
)

print("CNN model loaded successfully!")


# ============================================================
# IMAGE PREPROCESSING FUNCTION
# ============================================================

def preprocess_image(image):

    # --------------------------------------------------------
    # STEP 1: Convert image to grayscale
    # --------------------------------------------------------

    image = image.convert("L")

    # --------------------------------------------------------
    # STEP 2: Convert image to NumPy array
    # --------------------------------------------------------

    image_array = np.array(image)

    # --------------------------------------------------------
    # STEP 3: Determine whether background is dark or light
    #
    # We examine the border pixels because the border usually
    # represents the background.
    # --------------------------------------------------------

    border_pixels = np.concatenate([
        image_array[0, :],
        image_array[-1, :],
        image_array[:, 0],
        image_array[:, -1]
    ])

    border_mean = np.mean(border_pixels)

    # --------------------------------------------------------
    # MNIST format:
    # Background = black (0)
    # Digit      = white (255)
    #
    # If the border is bright, the image probably has:
    # white background + dark digit
    #
    # Therefore invert it.
    # --------------------------------------------------------

    if border_mean > 127:
        image_array = 255 - image_array

    # --------------------------------------------------------
    # STEP 4: Convert back to PIL image
    # --------------------------------------------------------

    image = Image.fromarray(
        image_array.astype(np.uint8)
    )

    # --------------------------------------------------------
    # STEP 5: Remove very small noise
    # --------------------------------------------------------

    image_array = np.array(image)

    # Anything below this threshold is considered background
    image_array[image_array < 30] = 0

    image = Image.fromarray(
        image_array.astype(np.uint8)
    )

    # --------------------------------------------------------
    # STEP 6: Find bounding box of the digit
    # --------------------------------------------------------

    bbox = image.getbbox()

    # If no digit is detected
    if bbox is None:

        # Return blank MNIST-style image
        blank = Image.new(
            "L",
            (28, 28),
            0
        )

        blank = np.array(blank).astype("float32") / 255.0

        return blank.reshape(
            1,
            28,
            28,
            1
        )

    # --------------------------------------------------------
    # STEP 7: Crop only the digit
    # --------------------------------------------------------

    image = image.crop(bbox)

    # --------------------------------------------------------
    # STEP 8: Preserve aspect ratio
    #
    # MNIST digits generally occupy around 20x20 pixels
    # inside the 28x28 image.
    # --------------------------------------------------------

    width, height = image.size

    target_size = 20

    if width > height:

        new_width = target_size

        new_height = max(
            1,
            int(height * target_size / width)
        )

    else:

        new_height = target_size

        new_width = max(
            1,
            int(width * target_size / height)
        )

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # STEP 9: Create 28x28 black canvas
    # --------------------------------------------------------

    canvas = Image.new(
        "L",
        (28, 28),
        0
    )

    # --------------------------------------------------------
    # STEP 10: Center the digit
    # --------------------------------------------------------

    x = (28 - new_width) // 2
    y = (28 - new_height) // 2

    canvas.paste(
        image,
        (x, y)
    )

    # --------------------------------------------------------
    # STEP 11: Convert to NumPy array
    # --------------------------------------------------------

    image_array = np.array(
        canvas
    ).astype("float32")

    # --------------------------------------------------------
    # STEP 12: Normalize
    # --------------------------------------------------------

    image_array = image_array / 255.0

    # --------------------------------------------------------
    # STEP 13: Reshape for CNN
    # --------------------------------------------------------

    image_array = image_array.reshape(
        1,
        28,
        28,
        1
    )

    return image_array


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ====================================================
        # CASE 1: UPLOADED IMAGE
        # ====================================================

        if "file" in request.files:

            file = request.files["file"]

            if file.filename == "":

                return jsonify({
                    "error":
                    "Please select an image."
                })

            image = Image.open(file)

        # ====================================================
        # CASE 2: DRAWING FROM CANVAS
        # ====================================================

        elif "image" in request.form:

            image_data = request.form["image"]

            # Remove data:image/png;base64,
            if "," in image_data:

                image_data = image_data.split(
                    ",",
                    1
                )[1]

            # Decode Base64
            image_bytes = base64.b64decode(
                image_data
            )

            image = Image.open(
                io.BytesIO(
                    image_bytes
                )
            )

        else:

            return jsonify({
                "error":
                "No image received."
            })

        # ====================================================
        # PREPROCESS IMAGE
        # ====================================================

        processed_image = preprocess_image(
            image
        )

        # ====================================================
        # CNN PREDICTION
        # ====================================================

        prediction = model.predict(
            processed_image,
            verbose=0
        )

        # ====================================================
        # FIND PREDICTED DIGIT
        # ====================================================

        predicted_digit = int(
            np.argmax(prediction[0])
        )

        # ====================================================
        # CALCULATE CONFIDENCE
        # ====================================================

        confidence = float(
            np.max(prediction[0]) * 100
        )

        # ====================================================
        # SEND RESULT TO FRONTEND
        # ====================================================

        return jsonify({

            "digit":
            predicted_digit,

            "confidence":
            round(
                confidence,
                2
            )

        })

    except Exception as e:

        print("Prediction error:", e)

        return jsonify({

            "error":
            str(e)

        })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )