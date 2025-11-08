from datetime import datetime
from app.core.mongo import db

Images = db["images"]

def insert_image_metadata(filename: str, width: int, height: int, source="user_upload"):
    doc = {
        "filename": filename,
        "width": width,
        "height": height,
        "uploaded_at": datetime.utcnow().isoformat(),
        "source": source,
        "detections_count": 0
    }
    result = Images.insert_one(doc)
    print(f"Inserted image metadata _id={result.inserted_id}")
    return result.inserted_id
