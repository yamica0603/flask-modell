from PIL import Image
import os
import numpy as np
from io import BytesIO
from keras.models import load_model
from keras.utils import img_to_array

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.h5")

if not os.path.exists(model_path):
    raise OSError(f"Non-existent model path, {model_path}")

model = load_model(model_path)


def preprocess_img(img_stream):
    try:
        img_stream = BytesIO(img_stream.read())
        print("Image stream read successfully")

        img = Image.open(img_stream)
        print("Image opened successfully")

        img = img.convert("RGB")
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array /= 255.0
        img_array = img_array.reshape(1, 224, 224, 3)

        return img_array
    except Exception as e:
        print(f"Error in preprocess_img: {str(e)}")
        return None


def predict_result(predict):
    pred = model.predict(predict)
    return np.argmax(pred[0], axis=-1)
