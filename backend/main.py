"""
backend/main.py
───────────────
FastAPI backend for the Gas Detection demo.

Endpoints:
  GET  /api/health   — liveness probe (wakes up Render cold start)
  GET  /api/random   — random sensor sample from pool → prediction
  POST /api/predict  — user-supplied sensor values → prediction

Deploy on Render:
  Build command : pip install -r requirements.txt
  Start command : uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os, json, pickle, random
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import onnxruntime as ort

# ─────────────────────────────────────────────────────────────
# Startup — load all artifacts once
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load(filename):
    return os.path.join(BASE_DIR, filename)

# ONNX inference session
ort_session = ort.InferenceSession(_load("model.onnx"))

# Scikit-learn preprocessors
with open(_load("scaler.pkl"), "rb") as f:
    scaler: object = pickle.load(f)

with open(_load("label_encoder.pkl"), "rb") as f:
    label_encoder: object = pickle.load(f)

CLASS_NAMES: list[str] = list(label_encoder.classes_)

# Sensor pool for random mode
with open(_load("sensor_pool.json")) as f:
    _pool_data   = json.load(f)
    SENSOR_POOL  = _pool_data["pool"]   # dict: class → list of [7 floats]

SENSOR_COLS = ["MQ2", "MQ3", "MQ5", "MQ6", "MQ7", "MQ8", "MQ135"]

# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="Gas Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tightened to your Vercel URL in production if needed
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────
def run_inference(raw_values: list[float]) -> dict:
    """
    raw_values : 7 raw sensor readings [MQ2, MQ3, MQ5, MQ6, MQ7, MQ8, MQ135]
    Returns    : dict with predicted_class, confidence, probabilities, sensors
    """
    arr        = np.array(raw_values, dtype="float32").reshape(1, -1)
    normalized = scaler.transform(arr).astype("float32")
    inp        = normalized.reshape(1, 1, 7)   # (batch=1, seq=1, sensors=7)

    logits = ort_session.run(None, {"sensor_input": inp})[0][0]   # shape (4,)
    # Stable softmax
    shifted = logits - logits.max()
    exp     = np.exp(shifted)
    probs   = exp / exp.sum()

    pred_idx   = int(probs.argmax())
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 1),
        "probabilities": {
            cls: round(float(p) * 100, 1)
            for cls, p in zip(CLASS_NAMES, probs)
        },
        "sensors": {col: round(float(v), 1) for col, v in zip(SENSOR_COLS, raw_values)},
    }

# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "classes": CLASS_NAMES}


@app.get("/api/random")
def random_sample():
    """
    Picks a random class, then a random sensor reading from that class's pool.
    Returns the reading alongside the model prediction.
    """
    true_class    = random.choice(CLASS_NAMES)
    raw_values    = random.choice(SENSOR_POOL[true_class])
    result        = run_inference(raw_values)
    result["true_class"] = true_class   # for display (may differ from prediction in rare cases)
    return result


class SensorInput(BaseModel):
    MQ2:   float = Field(..., ge=0, le=1023, description="MQ2 sensor reading")
    MQ3:   float = Field(..., ge=0, le=1023, description="MQ3 sensor reading")
    MQ5:   float = Field(..., ge=0, le=1023, description="MQ5 sensor reading")
    MQ6:   float = Field(..., ge=0, le=1023, description="MQ6 sensor reading")
    MQ7:   float = Field(..., ge=0, le=1023, description="MQ7 sensor reading")
    MQ8:   float = Field(..., ge=0, le=1023, description="MQ8 sensor reading")
    MQ135: float = Field(..., ge=0, le=1023, description="MQ135 sensor reading")


@app.post("/api/predict")
def predict(body: SensorInput):
    """
    Accepts 7 raw sensor values and returns the predicted gas class.
    """
    raw = [body.MQ2, body.MQ3, body.MQ5, body.MQ6, body.MQ7, body.MQ8, body.MQ135]
    try:
        return run_inference(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
