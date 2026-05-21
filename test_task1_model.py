import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILE = "task1_package.joblib"
model_path = os.path.join(MODEL_DIR, MODEL_FILE)

print("Loading Task 1 file from:")
print(model_path)

package = joblib.load(model_path)

print("\nTask 1 package loaded successfully.")
print("Package type:", type(package))

if isinstance(package, dict):
    print("Package keys:", package.keys())

task1_tfidf = package["tfidf"]
task1_model = package["model"]
task1_id2label = package["id2label"]

sample_text = (
    "The Company operates as a regional bank providing commercial banking, "
    "retail deposits, mortgage lending, credit cards, small business loans, "
    "and consumer financial services. "
    "Regional Banking Regional Banking Regional Banking "
    "Deposit accounts, commercial loans, mortgage banking, credit cards, "
    "consumer lending, and branch banking services."
)[:1000]

X = task1_tfidf.transform([sample_text])
pred_id = task1_model.predict(X)[0]

pred_label = task1_id2label.get(
    pred_id,
    task1_id2label.get(str(pred_id), str(pred_id))
)

print("\nPrediction ID:", pred_id)
print("Prediction Label:", pred_label)

if hasattr(task1_model, "predict_proba"):
    probs = task1_model.predict_proba(X)[0]
    print("Confidence:", round(float(np.max(probs)), 4))
elif hasattr(task1_model, "decision_function"):
    scores = task1_model.decision_function(X)
    print("Decision function shape:", scores.shape)
    print("Highest decision score:", float(np.max(scores)))
else:
   print("No probability or decision score available.")