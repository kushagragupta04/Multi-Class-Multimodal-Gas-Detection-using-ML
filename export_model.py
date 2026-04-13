"""
export_model.py
───────────────
One-time script: converts the trained SensorTransformer to ONNX format and
saves the fitted StandardScaler + LabelEncoder as pickle files.

Outputs (written to backend/):
  backend/model.onnx          — ONNX model for onnxruntime inference
  backend/scaler.pkl          — fitted StandardScaler
  backend/label_encoder.pkl   — fitted LabelEncoder

Run:
    python export_model.py
"""

import os, pickle, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
CSV_PATH        = os.path.join(
    SCRIPT_DIR,
    "Multimodal Gas Detection & Classification Dataset",
    "Gas Sensors Measurements",
    "Gas_Sensors_Measurements.csv",
)
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "transformer_best.pt")
BACKEND_DIR     = os.path.join(SCRIPT_DIR, "backend")
os.makedirs(BACKEND_DIR, exist_ok=True)

SENSOR_COLS = ["MQ2", "MQ3", "MQ5", "MQ6", "MQ7", "MQ8", "MQ135"]
SEED        = 42
D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT = 64, 4, 2, 128, 0.1

# ─────────────────────────────────────────────────────────────
# Fit preprocessors on the full dataset (same seed as training)
# ─────────────────────────────────────────────────────────────
print("Loading dataset ...")
df    = pd.read_csv(CSV_PATH)
X_raw = df[SENSOR_COLS].values.astype("float32")
y_raw = df["Gas"].values

label_enc   = LabelEncoder()
y_enc       = label_enc.fit_transform(y_raw)
class_names = label_enc.classes_
num_classes = len(class_names)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw).astype("float32")

print(f"  Classes : {list(class_names)}")
print(f"  Samples : {len(df)}")

# ─────────────────────────────────────────────────────────────
# SensorTransformer (identical to run_transformer.py)
# ─────────────────────────────────────────────────────────────
class SensorTransformer(nn.Module):
    def __init__(self, n_sensors=7, d_model=64, n_heads=4,
                 n_layers=2, d_ff=128, num_classes=4, dropout=0.1):
        super().__init__()
        self.input_proj   = nn.Linear(n_sensors, d_model)
        self.cls_token    = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoding = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        B   = x.size(0)
        x   = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_encoding
        x   = self.encoder(x)
        return self.head(x[:, 0])


# ─────────────────────────────────────────────────────────────
# Load trained weights
# ─────────────────────────────────────────────────────────────
print(f"\nLoading checkpoint: {os.path.basename(CHECKPOINT_PATH)}")
model = SensorTransformer(
    n_sensors=7, d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, d_ff=D_FF,
    num_classes=num_classes, dropout=DROPOUT,
)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
model.eval()

# ─────────────────────────────────────────────────────────────
# Export to ONNX
# ─────────────────────────────────────────────────────────────
onnx_path   = os.path.join(BACKEND_DIR, "model.onnx")
dummy_input = torch.zeros(1, 1, 7)   # (batch=1, seq=1, sensors=7)

print(f"\nExporting to ONNX → {onnx_path}")
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["sensor_input"],
    output_names=["logits"],
    dynamic_axes={"sensor_input": {0: "batch_size"}, "logits": {0: "batch_size"}},
)
print("  ONNX export done.")

# Quick sanity check with onnxruntime
try:
    import onnxruntime as ort
    sess    = ort.InferenceSession(onnx_path)
    test_in = np.zeros((1, 1, 7), dtype=np.float32)
    out     = sess.run(None, {"sensor_input": test_in})[0]
    print(f"  ONNX sanity check passed — output shape: {out.shape}")
except ImportError:
    print("  (onnxruntime not installed locally — install in backend env before deploying)")

# ─────────────────────────────────────────────────────────────
# Save preprocessors
# ─────────────────────────────────────────────────────────────
scaler_path = os.path.join(BACKEND_DIR, "scaler.pkl")
le_path     = os.path.join(BACKEND_DIR, "label_encoder.pkl")

with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"\nScaler saved      → {scaler_path}")

with open(le_path, "wb") as f:
    pickle.dump(label_enc, f)
print(f"LabelEncoder saved → {le_path}")

# ─────────────────────────────────────────────────────────────
# Save a pool of representative sensor samples (for /api/random)
# ─────────────────────────────────────────────────────────────
import json

sensor_pool = {}
for cls in class_names:
    # 8 evenly-spaced real samples per class (not the first few to avoid edge effects)
    mask    = y_raw == cls
    samples = df.loc[mask, SENSOR_COLS].values
    indices = np.linspace(50, len(samples) - 50, 8, dtype=int)
    sensor_pool[cls] = [samples[i].tolist() for i in indices]

pool_path = os.path.join(BACKEND_DIR, "sensor_pool.json")
with open(pool_path, "w") as f:
    json.dump({"classes": list(class_names), "pool": sensor_pool}, f, indent=2)
print(f"Sensor pool saved  → {pool_path}")

print("\nAll artifacts written to backend/. Ready to deploy.")
