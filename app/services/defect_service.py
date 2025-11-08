import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import threading
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ép TensorFlow không dùng GPU

model_defect = None  # Ban đầu chưa load


def load_defect_model():
    """Load model trong background thread để tránh chặn startup."""
    global model_defect
    try:
        model_defect = load_model("models/defect_model_fine_tuned.keras")
        print("✅ Defect model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Failed to load defect model: {e}")


# Gọi load sớm, không chặn khởi động FastAPI
threading.Thread(target=load_defect_model).start()

def predict_defect(img_path: str):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = float(model_defect.predict(img_array)[0][0])
    label = "normal" if prob >= 0.5 else "defect"
    return label, round(prob, 2)
