from datetime import datetime
from app.core.mongo import db

Detections = db["detections"]

def insert_detection(image_id, model_id, fruit_class, confidence, bbox):
    doc = {
        "image_id": image_id,
        "model_id": model_id,
        "fruit_class": fruit_class,
        "confidence": confidence,
        "bbox": bbox,
        "timestamp": datetime.utcnow().isoformat()
    }
    result = Detections.insert_one(doc)
    print(f"Inserted detection _id={result.inserted_id}")
    return result.inserted_id
