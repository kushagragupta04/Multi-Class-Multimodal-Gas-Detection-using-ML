"""
demo_inference.py
─────────────────
Demonstrates the SensorTransformer gas detection model end-to-end:
  1. Loads the dataset and fits the same preprocessors used during training
  2. Trains the model (or loads a saved checkpoint if one exists)
  3. Runs inference on representative dummy sensor readings for all 4 gas classes
  4. Displays formatted predictions with confidence scores

Run:
    python demo_inference.py
"""

import os, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
# Config  (identical to run_transformer.py)
# ─────────────────────────────────────────────────────────────
SEED       = 42
BATCH_SIZE = 64
LR         = 1e-3
NUM_EPOCHS = 80
PATIENCE   = 15
D_MODEL    = 64
N_HEADS    = 4
N_LAYERS   = 2
D_FF       = 128
DROPOUT    = 0.1

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
CSV_PATH        = os.path.join(
    SCRIPT_DIR,
    "Multimodal Gas Detection & Classification Dataset",
    "Gas Sensors Measurements",
    "Gas_Sensors_Measurements.csv",
)
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "transformer_best.pt")
SENSOR_COLS     = ["MQ2", "MQ3", "MQ5", "MQ6", "MQ7", "MQ8", "MQ135"]


# ─────────────────────────────────────────────────────────────
# SensorTransformer  (same architecture as run_transformer.py)
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B   = x.size(0)
        x   = self.input_proj(x)                    # (B, 1, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_encoding
        x   = self.encoder(x)
        return self.head(x[:, 0])                   # (B, num_classes)


# ─────────────────────────────────────────────────────────────
# Step 1 · Load data & fit preprocessors
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 68)
print("    GAS DETECTION SYSTEM  —  SensorTransformer DEMO")
print("═" * 68)
print(f"\n  Device : {device}")

print("\n  [1/3]  Loading dataset ...")
df = pd.read_csv(CSV_PATH)
print(f"         {len(df)} samples  |  classes: {sorted(df['Gas'].unique())}")

X_raw = df[SENSOR_COLS].values.astype(np.float32)
y_raw = df["Gas"].values

label_enc   = LabelEncoder()
y_enc       = label_enc.fit_transform(y_raw)
class_names = label_enc.classes_          # e.g. ['Mixture' 'NoGas' 'Perfume' 'Smoke']
num_classes = len(class_names)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

# One actual sample per class (index 100 is well into each class block,
# avoids edge samples while staying clearly within the class distribution)
class_samples = {
    cls: df.loc[y_raw == cls, SENSOR_COLS].iloc[100].values.astype(np.float32)
    for cls in class_names
}

# ─────────────────────────────────────────────────────────────
# Step 2 · Train model or load checkpoint
# ─────────────────────────────────────────────────────────────
X_tmp, X_te, y_tmp, y_te = train_test_split(
    X_scaled, y_enc, test_size=0.20, random_state=SEED, stratify=y_enc
)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_tmp, y_tmp, test_size=0.125, random_state=SEED, stratify=y_tmp
)

def make_loader(X, y, shuffle=False):
    Xt = torch.tensor(X).unsqueeze(1)          # (N, 1, 7)
    yt = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_tr, y_tr, shuffle=True)
val_loader   = make_loader(X_va, y_va)

model = SensorTransformer(
    n_sensors=7, d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, d_ff=D_FF,
    num_classes=num_classes, dropout=DROPOUT,
).to(device)

if os.path.exists(CHECKPOINT_PATH):
    print(f"\n  [2/3]  Loading checkpoint: {os.path.basename(CHECKPOINT_PATH)}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print("         Checkpoint loaded successfully — skipping training.")
else:
    print(f"\n  [2/3]  No checkpoint found. Training from scratch ...")
    print(f"         Max {NUM_EPOCHS} epochs, early stopping patience = {PATIENCE}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    best_val_acc   = 0.0
    best_state     = None
    patience_count = 0

    print(f"  {'Epoch':>6}  {'Train Acc':>9}  {'Val Acc':>8}")
    print("  " + "─" * 32)

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        tc, tt = 0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tc += (logits.argmax(1) == yb).sum().item()
            tt += Xb.size(0)
        scheduler.step()
        t_acc = tc / tt

        # Validate
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                vc += (model(Xb).argmax(1) == yb).sum().item()
                vt += Xb.size(0)
        v_acc = vc / vt

        if v_acc > best_val_acc:
            best_val_acc   = v_acc
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
            tag = "  ★ best"
        else:
            patience_count += 1
            tag = ""

        if epoch % 10 == 0 or tag:
            print(f"  {epoch:>6}  {t_acc*100:>8.2f}%  {v_acc*100:>7.2f}%{tag}")

        if patience_count >= PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, CHECKPOINT_PATH)
    print(f"\n  Training complete.  Best val accuracy: {best_val_acc*100:.2f}%")
    print(f"  Checkpoint saved → {CHECKPOINT_PATH}")


# ─────────────────────────────────────────────────────────────
# Step 3 · Demo Inference
# ─────────────────────────────────────────────────────────────
model.eval()

def predict(sensor_values):
    """
    sensor_values : array-like of 7 raw sensor readings
                    [MQ2, MQ3, MQ5, MQ6, MQ7, MQ8, MQ135]
    Returns       : (predicted_class, confidence_pct, all_probs)
    """
    raw   = np.array(sensor_values, dtype=np.float32).reshape(1, -1)
    norm  = scaler.transform(raw).astype(np.float32)
    x     = torch.tensor(norm).unsqueeze(1).to(device)   # (1, 1, 7)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = probs.argmax()
    return class_names[pred_idx], probs[pred_idx] * 100, probs


print("\n  [3/3]  Running inference on dummy sensor data ...")

# ── Section A : per-class dataset mean readings ──────────────
print("\n" + "─" * 68)
print("  [ A ]  Actual Dataset Samples  (one real sample per class)")
print("─" * 68)
print(f"  {'Expected':<10}  {'MQ2':>5} {'MQ3':>5} {'MQ5':>5} {'MQ6':>5} "
      f"{'MQ7':>5} {'MQ8':>5} {'MQ135':>5}    {'Predicted':<10}  {'Conf':>6}  OK?")
print("  " + "─" * 64)

for cls in class_names:
    vals              = class_samples[cls]
    pred, conf, probs = predict(vals)
    mark              = "✓" if pred == cls else "✗"
    vals_str          = " ".join(f"{v:>5.0f}" for v in vals)
    print(f"  {cls:<10}  {vals_str}    {pred:<10}  {conf:>5.1f}%   {mark}")

# ── Section B : custom handcrafted inputs ────────────────────
custom_tests = [
    # Values derived from actual dataset distribution ranges
    ("Smoke",   [620, 370, 340, 368, 580, 582, 308], "Typical combustion scenario"),
    ("Perfume", [753, 523, 489, 461, 685, 696, 495], "Typical perfume/alcohol vapor"),
    ("NoGas",   [748, 529, 416, 423, 615, 656, 458], "Typical clean air / baseline"),
    ("Mixture", [678, 442, 456, 418, 514, 371, 435], "Typical multi-gas mixture"),
]

print("\n" + "─" * 68)
print("  [ B ]  Custom Handcrafted Dummy Inputs")
print("─" * 68)
print(f"  {'Scenario':<32}  {'Predicted':<10}  {'Conf':>6}  OK?")
print("  " + "─" * 57)

for expected, vals, desc in custom_tests:
    pred, conf, probs = predict(vals)
    mark              = "✓" if pred == expected else "✗"
    print(f"  {desc:<32}  {pred:<10}  {conf:>5.1f}%   {mark}")

# ── Section C : full probability breakdown for one sample ────
print("\n" + "─" * 68)
print("  [ C ]  Full Probability Breakdown  (Smoke example)")
print("─" * 68)
_, _, probs = predict(class_samples["Smoke"])
for cls_name, prob in zip(class_names, probs):
    bar  = "█" * int(prob * 40)
    pad  = " " * (40 - len(bar))
    print(f"  {cls_name:<10}  {bar}{pad}  {prob*100:>5.1f}%")

print("\n" + "═" * 68)
print("    DEMO COMPLETE")
print("═" * 68 + "\n")
