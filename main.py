import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

# ── Load model artifacts ───────────────────────────────────────────────────────
try:
    model  = joblib.load('model_files/best_model.pkl')
    scaler = joblib.load('model_files/scaler.pkl')

    with open('model_files/feature_names.json') as f:
        FEATURES = json.load(f)

    with open('model_files/config.json') as f:
        CONFIG = json.load(f)

    THRESHOLD = CONFIG.get('optimal_threshold', 0.3)
    print(f"Model loaded successfully. Features: {len(FEATURES)}, Threshold: {THRESHOLD}")

except Exception as e:
    print(f"ERROR loading model: {e}")
    raise

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Network Intrusion Detection System",
    description="Detects cyberattacks in network traffic using ML",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema ─────────────────────────────────────────────────────────────
class NetworkFlow(BaseModel):
    features: Dict[str, float]

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status"    : "running",
        "model"     : CONFIG.get("model_name"),
        "accuracy"  : CONFIG.get("test_accuracy"),
        "roc_auc"   : CONFIG.get("roc_auc"),
        "threshold" : THRESHOLD
    }

@app.get("/health")
def health():
    return {"status": "healthy", "features_expected": len(FEATURES)}

@app.get("/features")
def get_features():
    return {"total": len(FEATURES), "feature_names": FEATURES}

@app.post("/predict")
def predict(flow: NetworkFlow):
    try:
        # Build input vector in correct feature order
        input_vector = np.array(
            [flow.features.get(f, 0.0) for f in FEATURES]
        ).reshape(1, -1)

        # Scale
        input_scaled = scaler.transform(input_vector)

        # Predict with optimal threshold
        attack_prob  = float(model.predict_proba(input_scaled)[0][1])
        prediction   = int(attack_prob >= THRESHOLD)

        return {
            "prediction" : prediction,
            "label"      : "ATTACK" if prediction == 1 else "NORMAL",
            "attack_prob": round(attack_prob * 100, 2),
            "normal_prob": round((1 - attack_prob) * 100, 2),
            "threshold"  : THRESHOLD,
            "confidence" : round(max(attack_prob, 1 - attack_prob) * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(flows: list[NetworkFlow]):
    if len(flows) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 flows per batch.")
    return [predict(flow) for flow in flows]