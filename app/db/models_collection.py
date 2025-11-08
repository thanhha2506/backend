from datetime import datetime
from app.core.mongo import db

Models = db["models"]

def insert_model_info(
    name,
    type,
    version,
    trained_on=None,
    metrics=None,
    framework=None,
    path=None,
    accuracy=None
):
    doc = {
        "name": name,
        "type": type,
        "version": version,
        "trained_on": trained_on,
        "framework": framework,
        "path": path,
        "metrics": metrics,
        "accuracy": accuracy,
        "date_trained": datetime.utcnow().isoformat()
    }
    result = Models.insert_one(doc)
    print(f"Inserted model info _id={result.inserted_id}")
    return result.inserted_id
