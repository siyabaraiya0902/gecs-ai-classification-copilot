from fastapi import FastAPI
from pydantic import BaseModel
import os
import csv
import joblib
import numpy as np
from scipy.sparse import hstack

app = FastAPI(title="Morningstar GECS Classification API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ----------------------------
# Load taxonomy lookup
# ----------------------------
TAXONOMY_PATH = os.path.join(DATA_DIR, "gecs_taxonomy_lookup.csv")


def load_taxonomy_lookup(path):
    lookup = {}

    if not os.path.exists(path):
        print("WARNING: taxonomy lookup file not found:", path)
        return lookup

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            code = str(row["code"]).strip()
            name = str(row["name"]).strip()
            level = str(row["level"]).strip()

            if code and name:
                lookup[code] = {
                    "name": name,
                    "level": level
                }

    return lookup


TAXONOMY_LOOKUP = load_taxonomy_lookup(TAXONOMY_PATH)


def display_label(code):
    code = str(code)
    item = TAXONOMY_LOOKUP.get(code)

    if item:
        return f"{item['name']} ({code})"

    return code


# ----------------------------
# Load Task 1 model package
# ----------------------------
TASK1_MODEL_PATH = os.path.join(MODEL_DIR, "task1_package.joblib")

task1_package = joblib.load(TASK1_MODEL_PATH)
task1_tfidf = task1_package["tfidf"]
task1_model = task1_package["model"]
task1_id2label = task1_package["id2label"]


# ----------------------------
# Load Task 2 model package
# ----------------------------
TASK2_MODEL_PATH = os.path.join(MODEL_DIR, "task2_pipeline.joblib")

task2_package = joblib.load(TASK2_MODEL_PATH)
optional_files = task2_package["optional_files"]

task2_model = optional_files["lr_t2.pkl"]
task2_tfidf_word = optional_files["tfidf_word_t2.pkl"]
task2_tfidf_char = optional_files["tfidf_char_t2.pkl"]
task2_id2label = task2_package["id2label_str"]


class PredictionRequest(BaseModel):
    long_profile: str
    segment_name: str
    segment_description: str


@app.get("/")
def home():
    return {
        "message": "Morningstar Task 1 + Task 2 model API is working",
        "taxonomy_codes_loaded": len(TAXONOMY_LOOKUP)
    }


def get_label(id2label, class_id):
    return id2label.get(
        class_id,
        id2label.get(str(class_id), str(class_id))
    )


def build_task1_text(long_profile: str, segment_name: str, segment_description: str):
    text = (
        long_profile + " " +
        segment_name + " " +
        segment_name + " " +
        segment_name + " " +
        segment_description
    )
    return text[:1000]


def build_task2_text(segment_name: str, segment_description: str):
    text = (
        segment_name + " " +
        segment_name + " " +
        segment_name + " " +
        segment_description
    )
    return text[:512]


def get_routing(confidence: float):
    if confidence >= 0.90:
        return "Auto approve or light review"
    elif confidence >= 0.70:
        return "Analyst review recommended"
    else:
        return "Manual classification required"


def predict_task1(long_profile: str, segment_name: str, segment_description: str):
    task1_text = build_task1_text(
        long_profile,
        segment_name,
        segment_description
    )

    X_task1 = task1_tfidf.transform([task1_text])

    pred_id = task1_model.predict(X_task1)[0]
    pred_code = get_label(task1_id2label, pred_id)
    pred_display = display_label(pred_code)

    if hasattr(task1_model, "decision_function"):
        scores = task1_model.decision_function(X_task1)

        if len(scores.shape) == 2:
            best_score = float(np.max(scores))
        else:
            best_score = float(scores[0])

        confidence_note = "Task 1 uses SVM decision score, not calibrated probability."
    else:
        best_score = None
        confidence_note = "Task 1 confidence unavailable."

    return {
        "predicted_industry": pred_display,
        "prediction_code": str(pred_code),
        "prediction_id": str(pred_id),
        "decision_score": best_score,
        "confidence_note": confidence_note
    }


def predict_task2(segment_name: str, segment_description: str):
    task2_text = build_task2_text(segment_name, segment_description)

    X_word = task2_tfidf_word.transform([task2_text])
    X_char = task2_tfidf_char.transform([task2_text])
    X_task2 = hstack([X_word, X_char])

    pred_id = task2_model.predict(X_task2)[0]
    pred_code = get_label(task2_id2label, pred_id)
    pred_display = display_label(pred_code)

    if hasattr(task2_model, "predict_proba"):
        probs = task2_model.predict_proba(X_task2)[0]
        classes = task2_model.classes_

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])

        top_indices = np.argsort(probs)[::-1][:3]

        top_alternatives = []
        for i in top_indices:
            class_id = classes[i]
            label_code = get_label(task2_id2label, class_id)
            label_display = display_label(label_code)

            top_alternatives.append({
                "label": label_display,
                "code": str(label_code),
                "confidence": round(float(probs[i]), 4)
            })
    else:
        confidence = 0.75
        top_alternatives = [
            {
                "label": pred_display,
                "code": str(pred_code),
                "confidence": confidence
            }
        ]

    return {
        "predicted_subindustry": pred_display,
        "prediction_code": str(pred_code),
        "prediction_id": str(pred_id),
        "confidence": confidence,
        "top_alternatives": top_alternatives
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    task1_result = predict_task1(
        request.long_profile,
        request.segment_name,
        request.segment_description
    )

    task2_result = predict_task2(
        request.segment_name,
        request.segment_description
    )

    routing = get_routing(task2_result["confidence"])

    return {
        "input_received": {
            "long_profile": request.long_profile,
            "segment_name": request.segment_name,
            "segment_description": request.segment_description
        },
        "predicted_industry": task1_result["predicted_industry"],
        "predicted_subindustry": task2_result["predicted_subindustry"],
        "confidence": task2_result["confidence"],
        "routing": routing,
        "task1_details": {
            "industry_prediction_code": task1_result["prediction_code"],
            "industry_prediction_id": task1_result["prediction_id"],
            "decision_score": task1_result["decision_score"],
            "confidence_note": task1_result["confidence_note"]
        },
        "task2_details": {
            "subindustry_prediction_code": task2_result["prediction_code"],
            "subindustry_prediction_id": task2_result["prediction_id"]
        },
        "evidence_card": {
            "reason": "Task 1 predicts broad industry from company and segment text. Task 2 predicts fine-grained activity/subindustry using word-level and character-level TF-IDF features. Predicted codes are translated using the GECS taxonomy lookup file.",
            "model_used": "Task 1 TF-IDF + Linear SVM; Task 2 Logistic Regression with word and character TF-IDF",
            "top_alternatives": task2_result["top_alternatives"]
        }
    }

    