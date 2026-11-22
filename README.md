# Handwritten Digit Recognition using CNN

An end-to-end deep learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. It includes model training, evaluation, and an interactive Streamlit application for predictions from uploaded images.

## Features
- Train a CNN on the MNIST handwritten digit dataset
- Evaluate the model on the held-out test set
- View accuracy, loss curves, and a confusion matrix
- Upload a handwritten digit image through a Streamlit interface
- Preprocess uploaded images to match the model's 28×28 grayscale input

## Project structure
    Handwritten_Image_recognition/
    ├── app.py
    ├── train.py
    ├── requirements.txt
    ├── .gitignore
    ├── README.md
    └── models/                 # Generated after training (not committed)

## 1. Create an environment
Python 3.10 or 3.11 is recommended.

    python -m venv .venv

Activate it:
- Windows: .venv\Scripts\activate
- macOS/Linux: source .venv/bin/activate

## 2. Install dependencies

    pip install -r requirements.txt

## 3. Train the model

    python train.py

The script downloads MNIST through Keras, trains the CNN, evaluates it on the test set, and saves the trained model to models/mnist_cnn.keras.

## 4. Run the web application
After training completes:

    streamlit run app.py

Open the local URL shown in your terminal, upload a clear image containing one handwritten digit, and click Predict digit.

## Image tips
- Upload a PNG, JPG, or JPEG image.
- Use one digit per image.
- A dark digit on a light background is recommended; the app automatically inverts the image when needed.
- Keep the digit centered and avoid heavy shadows or background patterns.

## Model architecture
The CNN uses two convolution + max-pooling blocks, followed by a dense layer and a 10-class softmax output. Training uses the MNIST training split and evaluates against the official test split.

## Important notes
- The trained model file is generated locally and is intentionally excluded from Git because it is a binary artifact.
- Run python train.py once before starting the app. The app displays a clear message if the model file is missing.
- MNIST contains centered, grayscale, 28×28 digit images. Predictions on photographs or unusual handwriting may be less accurate.

## Tech stack
Python · TensorFlow/Keras · NumPy · Matplotlib · Seaborn · scikit-learn · Streamlit · Pillow

## Future improvements
- Add a drawing canvas for direct digit input
- Compare CNN architectures and hyperparameters
- Add model export and deployment options

## License
This project is available under the MIT License. See LICENSE.
