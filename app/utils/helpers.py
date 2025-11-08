from app.db.images_collection import insert_image
from app.db.models_collection import insert_model
from app.db.detections_collection import insert_detection
from app.db.predictions_collection import insert_prediction

def save_full_pipeline(filename, width, height, model_info, yolo_result, ripeness, defect, final_state):
    image_id = insert_image(filename, width, height)
    model_id = insert_model(**model_info)
    detection_id = insert_detection(image_id, model_id, **yolo_result)
    insert_prediction(detection_id, yolo_result["fruit_class"], ripeness, defect, final_state, final_score=0.9)
