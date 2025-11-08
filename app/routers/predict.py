from fastapi import APIRouter, UploadFile, File
from datetime import datetime
from bson import ObjectId
import tempfile, os, time
from PIL import Image

# Import services
from app.services.yolo_service import predict_yolo
from app.services.defect_service import predict_defect
from app.services.ripeness_service import predict_ripeness

# Import DB operations
from app.db.images_collection import insert_image_metadata
from app.db.models_collection import insert_model_info
from app.db.detections_collection import insert_detection
from app.db.predictions_collection import insert_prediction

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/")
async def predict_pipeline(file: UploadFile = File(...)):
    """
    Nhận ảnh upload, chạy YOLO + 2 CNN, lưu vào MongoDB và trả kết quả tổng hợp.
    """

    # Kiểm tra file hợp lệ
    if not file.content_type.startswith("image/"):
        return {"error": "Uploaded file is not a valid image."}

    # Lưu file tạm (Windows-safe)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        await file.seek(0)
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp_path = tmp.name

    time.sleep(0.05)
    if not os.path.exists(tmp_path):
        return {"error": "Failed to save uploaded image."}

    #Mở ảnh an toàn
    try:
        img = Image.open(tmp_path)
    except Exception as e:
        return {"error": f"Cannot identify image: {str(e)}"}

    # Kiểm tra kích thước & dung lượng
    file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    width, height = img.size
    print(f"Uploaded {file.filename} ({file_size_mb:.2f} MB, {width}×{height}px)")

    MAX_SIZE = 1280
    if max(width, height) > MAX_SIZE:
        scale = MAX_SIZE / max(width, height)
        new_w, new_h = int(width * scale), int(height * scale)
        img_resized = img.resize((new_w, new_h))
        img_resized.save(tmp_path)
        width, height = new_w, new_h
        print(f"Resized to {new_w}×{new_h}px for faster YOLO inference.")

    if file_size_mb > 5:
        print("Warning: File too large (>5MB) — consider compressing before upload.")

    # Lưu metadata ảnh vào MongoDB
    image_id = insert_image_metadata(file.filename, width, height)
    print(f"Saved image metadata with _id = {image_id}")

    # Lưu thông tin model YOLO
    yolo_model_info = {
        "name": "YOLOv5",
        "type": "classifier",
        "version": "5.0",
        "framework": "PyTorch",
        "trained_on": "DeepFruit_Ensemble",
        "metrics": {"mAP": 0.88, "precision": 0.91, "recall": 0.86},
    }
    model_id = insert_model_info(**yolo_model_info)

    # Chạy YOLO
    yolo_result = predict_yolo(tmp_path)
    print(f" YOLO detected: {yolo_result}")

    fruit_class = yolo_result.get("fruit_class", "unknown")
    confidence = yolo_result.get("confidence", 0.0)
    bbox = yolo_result.get("bbox", None)

    if bbox is None:
        print(" YOLO: No bounding box found — likely classification-only model.")

    # Luôn lưu detection (bất kể kết quả)
    detection_id = insert_detection(
        image_id=image_id,
        model_id=model_id,
        fruit_class=fruit_class,
        confidence=confidence,
        bbox=bbox
    )

    # Kiểm tra loại trái cây
    VALID_FRUITS = {"mango", "apple", "papaya"}

    # Nếu confidence thấp hoặc loại trái cây không hợp lệ → dừng pipeline
    if (
            fruit_class is None
            or fruit_class.lower() not in VALID_FRUITS
            or confidence == 0.0
    ):
        print(
            f" '{fruit_class}' có confidence = {confidence} hoặc không thuộc {VALID_FRUITS}. Dừng pipeline tại YOLO.")
        confidence = 0.0

        detection_id = insert_detection(
            image_id=image_id,
            model_id=model_id,
            fruit_class=fruit_class,
            confidence=confidence,
            bbox=bbox
        )

        prediction_id = insert_prediction(
            detection_id=detection_id,
            fruit_class=fruit_class,
            ripeness={
                "label": None,
                "confidence": 0.0,
                "model_id": None
            },
            defect={
                "label": None,
                "confidence": 0.0,
                "model_id": None
            },
            final_state="unidentified",
            final_score=0.0
        )

        return {
            "filename": file.filename,
            "uploaded_at": datetime.utcnow().isoformat(),
            "result": {
                "fruit_type": fruit_class,
                "confidence": confidence,
                "final_state": "unidentified",
                "final_score": 0.0,
                "message": f"'{fruit_class}' không đạt confidence đủ cao hoặc không phải xoài, táo, đu đủ. Bỏ qua defect/ripeness."
            }
        }

    # Nếu là trái cây hợp lệ → chạy CNN
    print(f"'{fruit_class}' hợp lệ — tiến hành đánh giá defect và ripeness...")

    defect_label, defect_conf = predict_defect(tmp_path)
    ripeness_label, ripeness_conf = predict_ripeness(tmp_path)

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

    final_state = (
        "harvestable" if ripeness_label == "ripe" and defect_label == "normal"
        else "not_harvestable"
    )
    final_score = round((ripeness_conf + defect_conf) / 2, 2)

    insert_prediction(
        detection_id=detection_id,
        fruit_class=fruit_class,
        ripeness=ripeness_obj,
        defect=defect_obj,
        final_state=final_state,
        final_score=final_score
    )

    #  Trả kết quả cuối cùng
    return {
        "filename": file.filename,
        "uploaded_at": datetime.utcnow().isoformat(),
        "result": {
            "fruit_type": fruit_class,
            "confidence": confidence,
            "defect_status": defect_label,
            "ripeness_status": ripeness_label,
            "final_state": final_state,
            "final_score": final_score
        }
    }
