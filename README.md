# Handwritten Digit Recognition Using Custom CNN

## Project Overview

This project focuses on recognizing handwritten digits (0–9) using a custom Convolutional Neural Network (CNN). The model is trained and evaluated using the MNIST dataset and integrated into a Flask-based web application.

Users can draw a digit directly on the canvas or upload a handwritten digit image. The trained CNN predicts the digit and displays the prediction confidence.

## Objectives

- Study handwritten digit recognition using deep learning.
- Use the MNIST dataset for training and testing.
- Design and implement a custom CNN architecture.
- Preprocess handwritten digit images.
- Train and evaluate the CNN model.
- Develop a web application for digit prediction.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Pillow (PIL) | Image processing |
| Flask | Backend web framework |
| HTML | Web page structure |
| CSS | Web page styling |
| JavaScript | Canvas drawing and frontend interaction |

## Dataset

The project uses the **MNIST handwritten digit dataset**.

- Training images: 60,000
- Testing images: 10,000
- Number of classes: 10 (digits 0–9)
- Image dimensions: 28 × 28 pixels
- Image format: Grayscale

Pixel values are normalized from 0–255 to 0–1 before training.

## Custom CNN Architecture

The CNN model consists of the following layers:

1. **Conv2D:** 32 filters with a 3 × 3 kernel and ReLU activation.
2. **MaxPooling2D:** 2 × 2 pooling.
3. **Conv2D:** 64 filters with a 3 × 3 kernel and ReLU activation.
4. **MaxPooling2D:** 2 × 2 pooling.
5. **Flatten:** Converts feature maps into a one-dimensional vector.
6. **Dense:** 128 neurons with ReLU activation.
7. **Dropout:** 0.5 dropout rate to reduce overfitting.
8. **Output Dense:** 10 neurons with Softmax activation for digit classification.

### Model Training Configuration

- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Epochs: 10
- Batch size: 64
- Validation split: 10%

## Project Structure

```text
MNIST_CNN_Project/
│
├── app.py
├── train_model.py
├── test_model.py
├── mnist_cnn_model.keras
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/jhansiraniyc-del/MNIST-CNN-Handwritten-Digit-Recognition.git
```

### 2. Navigate to the Project Folder

```bash
cd MNIST-CNN-Handwritten-Digit-Recognition
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python app.py
```

### 7. Open in Browser

Visit:

```text
http://127.0.0.1:5000
```

## How to Use

1. Open the web application.
2. Draw a handwritten digit (0–9) on the canvas or upload an image.
3. Click the **Predict** button.
4. The application processes the image and sends it to the trained CNN model.
5. The predicted digit and confidence score are displayed.

**Note:** The current model is designed to recognize one handwritten digit per input image. It does not directly recognize multiple-digit numbers.

## Model Evaluation

The model is evaluated using the MNIST testing dataset. Training and validation accuracy and loss are also visualized using Matplotlib.

The actual test accuracy is obtained by running `train_model.py` or evaluating the saved model using `test_model.py`.

## Future Enhancements

- Support recognition of multiple handwritten digits.
- Improve prediction accuracy for different handwriting styles.
- Add a prediction history feature.
- Develop a mobile-friendly interface.
- Deploy the application online.

## Author

**Jhansirani Y C**

Information Science and Engineering

GitHub: [jhansiraniyc-del](https://github.com/jhansiraniyc-del)

## License

This project is developed for academic and educational purposes.
