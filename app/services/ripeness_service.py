import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import threading
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ép TensorFlow không dùng GPU

model_ripeness = None

def load_ripeness_model():
    """Load model trong background thread."""
    global model_ripeness
    try:
        model_ripeness = load_model("models/ripeness_model_fine_tuned.keras")
        print("✅ Ripeness model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Failed to load ripeness model: {e}")
    
# Load sớm không chặn FastAPI
threading.Thread(target=load_ripeness_model).start()

def predict_ripeness(img_path: str):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = float(model_ripeness.predict(img_array)[0][0])
    label = "ripe" if prob >= 0.5 else "unripe"
    return label, round(prob, 2)
