import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "FineTuned_DenseNet221[0.94486].h5")


model = load_model(MODEL_PATH)


CLASS_NAMES = ["Normal", "Pneumonia", "TB"]

def predict_xray(img_file):
    """
    Predict the class of a chest X-ray image.

    Parameters
    ----------
    img_file : file-like object
        Uploaded image file.

    Returns
    -------
    tuple
        predicted_label (str), probabilities (numpy array)
    """
    img = image.load_img(img_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0  # normalize to 0–1
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], preds
