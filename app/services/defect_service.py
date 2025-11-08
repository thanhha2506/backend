import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model_defect = load_model("models/defect_model_fine_tuned.keras")

def predict_defect(img_path: str):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = float(model_defect.predict(img_array)[0][0])
    label = "normal" if prob >= 0.5 else "defect"
    return label, round(prob, 2)
