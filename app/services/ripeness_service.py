import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model_ripeness = load_model("models/ripeness_model_fine_tuned.keras")

def predict_ripeness(img_path: str):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = float(model_ripeness.predict(img_array)[0][0])
    label = "ripe" if prob >= 0.5 else "unripe"
    return label, round(prob, 2)
