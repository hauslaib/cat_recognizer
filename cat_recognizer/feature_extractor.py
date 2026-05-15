"""VGG16 feature extractor shared by training and inference."""
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

_model = None


def get_model():
    global _model
    if _model is None:
        base = VGG16(weights='imagenet')
        _model = Model(inputs=base.input, outputs=base.get_layer('fc2').output)
    return _model


def extract_features(img_path):
    """Return a 4096-dim feature vector for the image at img_path."""
    m = get_model()
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return m.predict(x, verbose=0).flatten()


def l2_normalize(vec, eps=1e-9):
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / (norm + eps)
