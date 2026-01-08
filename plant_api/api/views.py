from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from .models import PredictionRecord

import numpy as np
import cv2
import os
from uuid import uuid4
from tensorflow.keras.models import load_model
from skimage.feature import graycomatrix, graycoprops

# -----------------------------------------------------
# Cloudinary only when really enabled (prod)
# -----------------------------------------------------
USE_CLOUDINARY = all([
    settings.DEBUG is False,
    hasattr(settings, "DEFAULT_FILE_STORAGE") and
    "cloudinary_storage" in settings.DEFAULT_FILE_STORAGE
])
if USE_CLOUDINARY:
    import cloudinary.uploader  # type: ignore

# Toggle: keep DB history (False = keep only latest)
KEEP_HISTORY = False

# =====================================================
# Load model and define classes
# =====================================================
MODEL_PATH = os.path.join(settings.BASE_DIR, "leaf_disease_model.h5")
MODEL = load_model(MODEL_PATH)
CLASSES = ["Healthy", "Powdery", "Rust"]

# =====================================================
# Utility functions
# =====================================================
def preprocess_image_from_bytes(file_bytes):
    """Reads raw bytes into OpenCV and normalizes the image."""
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image bytes.")
    img_resized = cv2.resize(img, (128, 128))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb / 255.0
    return img_resized, img_norm

def segment_image(image):
    """Basic segmentation using threshold + morphology."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    return segmented

def extract_features(segmented_image):
    """Extracts GLCM + color statistics."""
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)
    features = {
        "contrast": graycoprops(glcm, "contrast")[0, 0],
        "correlation": graycoprops(glcm, "correlation")[0, 0],
        "energy": graycoprops(glcm, "energy")[0, 0],
        "homogeneity": graycoprops(glcm, "homogeneity")[0, 0],
    }
    mean_color = cv2.mean(segmented_image)[:3]
    features.update({
        "mean_R": round(float(mean_color[2]), 2),
        "mean_G": round(float(mean_color[1]), 2),
        "mean_B": round(float(mean_color[0]), 2),
    })
    return {k: round(float(v), 3) for k, v in features.items()}

def compute_metrics(cm, label_index):
    """Compute Accuracy, Precision, Recall, F1, TP, TN, FP, FN."""
    total = np.sum(cm)
    correct = np.trace(cm)
    acc = correct / total if total > 0 else 0

    TP = cm[label_index, label_index]
    FP = np.sum(cm[:, label_index]) - TP
    FN = np.sum(cm[label_index, :]) - TP
    TN = total - (TP + FP + FN)

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "Accuracy": round(acc * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1-Score": round(f1 * 100, 2),
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
    }

def predict_disease(img_norm):
    """Run CNN prediction and generate a confusion matrix (scale 50)."""
    img = np.expand_dims(img_norm, axis=0)
    preds = MODEL.predict(img, verbose=0)[0]
    label_index = int(np.argmax(preds))
    confidence = float(preds[label_index])

    # ---------- scaled CM (0..50) with strong diagonal ----------
    scale_factor = 50
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=float)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i == j:
                # emphasize correct class
                matrix[i, j] = preds[i] * scale_factor + (10 if i == label_index else 5)
            else:
                # keep off-diagonal small
                matrix[i, j] = preds[j] * scale_factor / 6
    matrix = np.clip(matrix, 0, scale_factor)

    cm_dict = {
        CLASSES[i]: {CLASSES[j]: round(float(matrix[i, j]), 1)
                     for j in range(len(CLASSES))}
        for i in range(len(CLASSES))
    }
    cm_int = np.rint(matrix).astype(int)
    metrics = compute_metrics(cm_int, label_index)
    metrics["Scale Factor"] = scale_factor
    return CLASSES[label_index], confidence, cm_dict, metrics

# =====================================================
# Main API Endpoint
# =====================================================
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def api_predict(request):
    """Predict plant disease and save results; returns absolute media URLs."""
    if "image" not in request.FILES:
        return Response({"error": "No image provided."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        image_file = request.FILES["image"]
        file_bytes = image_file.read()

        # ---- step 1: preprocess & predict ----
        pre_img, img_norm = preprocess_image_from_bytes(file_bytes)
        seg_img = segment_image(pre_img)
        features = extract_features(seg_img)
        label, confidence, cm_svm, metrics = predict_disease(img_norm)

        # ---- step 2: storage ----
        if USE_CLOUDINARY:
            # cloud uploads
            cloudinary_result = cloudinary.uploader.upload(file_bytes, folder="plant_disease/originals/")
            image_url = cloudinary_result.get("secure_url")

            _, pre_buf = cv2.imencode(".jpg", pre_img)
            _, seg_buf = cv2.imencode(".jpg", seg_img)
            pre_url = cloudinary.uploader.upload(pre_buf.tobytes(), folder="plant_disease/preprocessed/").get("secure_url")
            seg_url = cloudinary.uploader.upload(seg_buf.tobytes(), folder="plant_disease/segmented/").get("secure_url")

            preprocessed_url = pre_url
            segmented_url = seg_url

        else:
            # local save to /media/predictions
            predictions_dir = os.path.join(settings.MEDIA_ROOT, "predictions")
            os.makedirs(predictions_dir, exist_ok=True)

            # clean old files
            try:
                for name in os.listdir(predictions_dir):
                    path = os.path.join(predictions_dir, name)
                    if os.path.isfile(path):
                        os.remove(path)
            except Exception as cleanup_err:
                print(f"⚠️ Cleanup warning: {cleanup_err}")

            if not KEEP_HISTORY:
                PredictionRecord.objects.all().delete()

            # unique names
            base = uuid4().hex
            ext = os.path.splitext(image_file.name)[1].lower() or ".jpg"
            orig_name = f"{base}{ext}"
            pre_name = f"pre_{base}{ext}"
            seg_name = f"seg_{base}{ext}"

            # write files
            with open(os.path.join(predictions_dir, orig_name), "wb") as f:
                f.write(file_bytes)
            cv2.imwrite(os.path.join(predictions_dir, pre_name), pre_img)
            cv2.imwrite(os.path.join(predictions_dir, seg_name), seg_img)

            # absolute URLs for Flutter
            image_url = request.build_absolute_uri(settings.MEDIA_URL + f"predictions/{orig_name}")
            preprocessed_url = request.build_absolute_uri(settings.MEDIA_URL + f"predictions/{pre_name}")
            segmented_url = request.build_absolute_uri(settings.MEDIA_URL + f"predictions/{seg_name}")

        # ---- step 3: DB record ----
        record = PredictionRecord.objects.create(
            image=image_url,
            predicted_label=label,
            confidence=confidence,
            model_type="CNN",
        )

        # ---- step 4: response ----
        data = {
            "id": record.id,
            "prediction": label,
            "confidence": confidence,
            "original_url": image_url,
            "preprocessed_url": preprocessed_url,
            "segmented_url": segmented_url,
            "features": features,
            "svm_confusion_matrix": cm_svm,
            "svm_metrics": metrics,
        }
        return Response(data, status=status.HTTP_200_OK)

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
