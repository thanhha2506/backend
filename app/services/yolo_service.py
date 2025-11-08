# app/services/yolo_service.py
import torch
import cv2
import numpy as np
from PIL import Image
from bson import ObjectId

from app.services.defect_service import predict_defect
from app.services.ripeness_service import predict_ripeness
from app.db.predictions_collection import insert_prediction
from app.db.images_collection import insert_image_metadata
from app.db.detections_collection import insert_detection
from app.db.models_collection import insert_model_info
from app.core.mongo import Models

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device


device = select_device("cpu")
model_yolo = attempt_load("models/best.pt", device=device)
model_yolo.eval()

model_info = {
    "name": "YOLOv5 Fruit Detector",
    "type": "detector",
    "version": "v5.0",
    "trained_on": "DeepFruit_Ensemble",
    "metrics": {"mAP": 0.88, "precision": 0.91, "recall": 0.86},
}
existing = Models.find_one({"name": model_info["name"], "version": model_info["version"]})
if not existing:
    insert_model_info(**model_info)
else:
    print("Model metadata already exists in MongoDB.")


# === HÀM CHÍNH: predict_yolo ===
def predict_yolo(image_path: str):
    # 1. Đọc ảnh và lưu metadata vào MongoDB
    img = Image.open(image_path)
    width, height = img.size
    image_id = insert_image_metadata(
        filename=image_path.split("/")[-1],
        width=width,
        height=height
    )

    # 2. Lấy model_id YOLO
    model_doc = Models.find_one({"name": "YOLOv5 Fruit Detector"})
    model_id = model_doc["_id"]

    # 3. Chạy YOLO detect
    img_cv = cv2.imread(image_path)
    img_resized = cv2.resize(img_cv, (640, 640))
    img_input = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_input = img_input.unsqueeze(0).to(device)

    pred = model_yolo(img_input)[0]
    detections = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    if detections is None or len(detections) == 0:
        return {
            "fruit_class": None,
            "confidence": 0.0,
            "bbox": None,
            "skipped_cnn": True,
            "reason": "No object detected"
        }

    x1, y1, x2, y2, conf, cls = detections[0].cpu().numpy()
    fruit_class = model_yolo.names[int(cls)]
    confidence = float(conf)
    bbox = {"x_min": int(x1), "y_min": int(y1), "x_max": int(x2), "y_max": int(y2)}

    detection_id = insert_detection(image_id, model_id, fruit_class, confidence, bbox)

    # 4. Kiểm tra logic bỏ qua CNN
    valid_classes = ["mango", "apple", "papaya"]
    if (fruit_class not in valid_classes) or (confidence < 0.3):
        # Lưu prediction tạm thời (chưa chạy CNN)
        prediction_id = insert_prediction(
            detection_id=detection_id,
            fruit_class=fruit_class,
            ripeness={"label": None, "confidence": 0.0, "model_id": None},
            defect={"label": None, "confidence": 0.0, "model_id": None},
            final_state="unknown",
            final_score=0.0
        )
        return {
            "image_id": str(image_id),
            "detection_id": str(detection_id),
            "prediction_id": str(prediction_id),
            "fruit_class": fruit_class,
            "confidence": confidence,
            "skipped_cnn": True,
            "reason": f"Skipped CNN because fruit_class='{fruit_class}' or confidence={confidence:.2f} < 0.3"
        }

    # 5. Chạy song song 2 model CNN
    defect_label, defect_conf = predict_defect(image_path)
    ripeness_label, ripeness_conf = predict_ripeness(image_path)

    ripeness_obj = {
        "label": ripeness_label,
        "confidence": ripeness_conf,
        "model_id": ObjectId()
    }
    defect_obj = {
        "label": defect_label,
        "confidence": defect_conf,
        "model_id": ObjectId()
    }

    # 6. Tổng hợp kết quả cuối
    final_state = (
        "harvestable"
        if ripeness_label == "ripe" and defect_label == "normal"
        else "not_harvestable"
    )
    final_score = round((ripeness_conf + defect_conf) / 2, 2)

    prediction_id = insert_prediction(
        detection_id=detection_id,
        fruit_class=fruit_class,
        ripeness=ripeness_obj,
        defect=defect_obj,
        final_state=final_state,
        final_score=final_score,
    )

    # 7. Trả về kết quả JSON hoàn chỉnh
    return {
        "image_id": str(image_id),
        "detection_id": str(detection_id),
        "prediction_id": str(prediction_id),
        "fruit_class": fruit_class,
        "confidence": confidence,
        "defect_status": defect_label,
        "ripeness_status": ripeness_label,
        "final_state": final_state,
        "final_score": final_score,
        "skipped_cnn": False
    }