from datetime import datetime
from app.core.mongo import db

Predictions = db["predictions"]

def insert_prediction(detection_id, fruit_class, ripeness, defect, final_state, final_score):
    doc = {
        "detection_id": detection_id,
        "fruit_class": fruit_class,
        "ripeness": ripeness,
        "defect": defect,
        "final_state": final_state,
        "final_score": final_score,
        "timestamp": datetime.utcnow().isoformat()
    }
    result = Predictions.insert_one(doc)
    print(f"Inserted prediction _id={result.inserted_id}")
    return result.inserted_id
