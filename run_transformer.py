import os, sys, time, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
LINE  = "─" * 65
DLINE = "═" * 65

def banner(title: str, char="═"):
    bar = char * 65
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)

def step(msg: str):
    print(f"\n  ▶  {msg}")


# ─────────────────────────────────────────────────────────────
# 0. Config
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

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(
    SCRIPT_DIR,
    "Multimodal Gas Detection & Classification Dataset",
    "Gas Sensors Measurements",
    "Gas_Sensors_Measurements.csv"
)

banner("SensorTransformer — Gas Detection (Gen AI)")
print(f"  Device      : {device}")
print(f"  Batch size  : {BATCH_SIZE}")
print(f"  Max epochs  : {NUM_EPOCHS}  |  Patience: {PATIENCE}")
print(f"  Architecture: {N_LAYERS} layers, {N_HEADS} heads, d={D_MODEL}, d_ff={D_FF}")


# ─────────────────────────────────────────────────────────────
# 1. Load & Preprocess Data
# ─────────────────────────────────────────────────────────────
banner("1 · Loading Dataset", "─")
step(f"Reading CSV: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"  Shape       : {df.shape}")
print(f"  Columns     : {list(df.columns)}")
print(f"  Class counts:\n{df['Gas'].value_counts().to_string()}")

SENSOR_COLS = ["MQ2", "MQ3", "MQ5", "MQ6", "MQ7", "MQ8", "MQ135"]
TARGET_COL  = "Gas"

X_raw = df[SENSOR_COLS].values.astype(np.float32)
y_raw = df[TARGET_COL].values

label_enc   = LabelEncoder()
y_enc       = label_enc.fit_transform(y_raw)
class_names = label_enc.classes_
num_classes = len(class_names)

step(f"Classes ({num_classes}): {list(class_names)}")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

# 70 / 10 / 20 split
X_tmp, X_te, y_tmp, y_te = train_test_split(
    X_scaled, y_enc, test_size=0.20, random_state=SEED, stratify=y_enc
)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_tmp, y_tmp, test_size=0.125, random_state=SEED, stratify=y_tmp
)
step(f"Split — Train: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}")

def make_loader(X, y, shuffle=False):
    Xt = torch.tensor(X).unsqueeze(1)   # (N, 1, 7)
    yt = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_tr, y_tr, shuffle=True)
val_loader   = make_loader(X_va, y_va)
test_loader  = make_loader(X_te, y_te)


# ─────────────────────────────────────────────────────────────
# 2. Model Definition
# ─────────────────────────────────────────────────────────────
banner("2 · Building SensorTransformer", "─")

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
        x   = self.input_proj(x)                            # (B,1,d)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_encoding
        x   = self.encoder(x)
        return self.head(x[:, 0])


model        = SensorTransformer(
    n_sensors=7, d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, d_ff=D_FF, num_classes=num_classes, dropout=DROPOUT
).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable parameters: {total_params:,}")


# ─────────────────────────────────────────────────────────────
# 3. Training
# ─────────────────────────────────────────────────────────────
banner("3 · Training", "─")

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
best_val_acc   = 0.0
patience_count = 0
best_state     = None
t0             = time.time()

header = f"  {'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}  {'LR':>8}"
print(header)
print("  " + LINE)

for epoch in range(1, NUM_EPOCHS + 1):
    # ── Train ──
    model.train()
    tl, tc, tt = 0.0, 0, 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tl += loss.item() * Xb.size(0)
        tc += (logits.argmax(1) == yb).sum().item()
        tt += Xb.size(0)
    scheduler.step()
    t_loss = tl / tt;  t_acc = tc / tt

    # ── Validate ──
    model.eval()
    vl, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss   = criterion(logits, yb)
            vl += loss.item() * Xb.size(0)
            vc += (logits.argmax(1) == yb).sum().item()
            vt += Xb.size(0)
    v_loss = vl / vt;  v_acc = vc / vt

    train_losses.append(t_loss); val_losses.append(v_loss)
    train_accs.append(t_acc);   val_accs.append(v_acc)

    # Early stopping
    if v_acc > best_val_acc:
        best_val_acc   = v_acc
        best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        patience_count = 0
        tag = " ★"
    else:
        patience_count += 1
        tag = ""
        if patience_count >= PATIENCE:
            print(f"\n  ⏹  Early stopping at epoch {epoch}  "
                  f"(no val improvement for {PATIENCE} epochs)")
            break

    cur_lr = scheduler.get_last_lr()[0]
    # Print every epoch
    print(
        f"  {epoch:>6}  {t_loss:>10.4f}  {t_acc*100:>8.2f}%  "
        f"{v_loss:>9.4f}  {v_acc*100:>7.2f}%  {cur_lr:>8.2e}{tag}"
    )

elapsed = time.time() - t0
print(f"\n  Training complete in {elapsed:.1f}s")
print(f"  Best validation accuracy: {best_val_acc*100:.2f}%")
model.load_state_dict(best_state)


# ─────────────────────────────────────────────────────────────
# 4. Test Evaluation
# ─────────────────────────────────────────────────────────────
banner("4 · Test Set Evaluation", "─")

model.eval()
preds, labels = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        preds.extend(model(Xb.to(device)).argmax(1).cpu().numpy())
        labels.extend(yb.numpy())

preds  = np.array(preds)
labels = np.array(labels)
acc    = accuracy_score(labels, preds)

print(f"\n  ✅  Test Accuracy : {acc*100:.2f}%\n")
print(classification_report(labels, preds, target_names=class_names))


# ─────────────────────────────────────────────────────────────
# 5. Sensor Importance (Gradient Attribution)
# ─────────────────────────────────────────────────────────────
banner("5 · Sensor Importance (Gradient Attribution)", "─")

model.eval()
sample_X, sample_y = next(iter(test_loader))
inp = sample_X[:16].to(device).requires_grad_(True)
model(inp).mean().backward()
importance = inp.grad.abs().squeeze(1).mean(0).cpu().detach().numpy()
importance = importance / importance.sum() * 100   # as %

print(f"\n  {'Sensor':<10}  {'Importance':>10}  Bar")
print("  " + "─" * 40)
max_i = importance.max()
for sensor, imp in zip(SENSOR_COLS, importance):
    bar = "█" * int(imp / max_i * 30)
    print(f"  {sensor:<10}  {imp:>9.2f}%  {bar}")


# ─────────────────────────────────────────────────────────────
# 6. Final Comparison Table
# ─────────────────────────────────────────────────────────────
banner("6 · Model Comparison", "─")

comparison = [
    ("Logistic Regression",         83.00, "Classical ML"),
    ("Support Vector Machine",      91.00, "Classical ML"),
    ("Random Forest",               96.93, "Ensemble ML"),
    ("K-Nearest Neighbors",         97.08, "Instance ML"),
    ("SensorTransformer (Gen AI)",  acc * 100, "Gen AI / Deep Learning"),
]
comparison.sort(key=lambda r: r[1], reverse=True)

print(f"\n  {'Rank':<5}  {'Model':<35}  {'Accuracy':>9}  {'Type'}")
print("  " + "─" * 75)
for rank, (name, accuracy, mtype) in enumerate(comparison, 1):
    star = "🏆" if rank == 1 else f" {rank} "
    print(f"  {star}    {name:<35}  {accuracy:>8.2f}%  {mtype}")


# ─────────────────────────────────────────────────────────────
# 7. Save Plots
# ─────────────────────────────────────────────────────────────
banner("7 · Saving Plots", "─")

# Learning curves
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(train_losses, label="Train Loss", color="#3b82f6", lw=2)
axes[0].plot(val_losses,   label="Val Loss",   color="#f59e0b", lw=2)
axes[0].set(title="Loss Curves", xlabel="Epoch", ylabel="Cross-Entropy Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot([a*100 for a in train_accs], label="Train Acc", color="#10b981", lw=2)
axes[1].plot([a*100 for a in val_accs],   label="Val Acc",   color="#8b5cf6", lw=2)
axes[1].set(title="Accuracy Curves", xlabel="Epoch", ylabel="Accuracy (%)")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.suptitle("SensorTransformer — Learning Curves", fontsize=14, fontweight="bold")
plt.tight_layout()
out_curves = os.path.join(SCRIPT_DIR, "transformer_learning_curves.png")
plt.savefig(out_curves, dpi=150, bbox_inches="tight")
plt.close()
step(f"Saved → {out_curves}")

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, ax=ax)
ax.set(title=f"SensorTransformer — Confusion Matrix  (Acc: {acc*100:.2f}%)",
       ylabel="True Label", xlabel="Predicted Label")
plt.tight_layout()
out_cm = os.path.join(SCRIPT_DIR, "transformer_confusion_matrix.png")
plt.savefig(out_cm, dpi=150, bbox_inches="tight")
plt.close()
step(f"Saved → {out_cm}")

# Sensor importance bar chart
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#ec4899","#14b8a6"]
ax.bar(SENSOR_COLS, importance, color=colors, edgecolor="white", lw=1.2)
for i, (s, v) in enumerate(zip(SENSOR_COLS, importance)):
    ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set(title="Sensor Importance (Gradient Attribution)",
       ylabel="Normalized Importance (%)", xlabel="Gas Sensor")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out_imp = os.path.join(SCRIPT_DIR, "transformer_sensor_importance.png")
plt.savefig(out_imp, dpi=150, bbox_inches="tight")
plt.close()
step(f"Saved → {out_imp}")

banner("Done ✓")
print(f"  Final test accuracy : {acc*100:.2f}%")
print(f"  Best val accuracy   : {best_val_acc*100:.2f}%")
print(f"  Plots saved to      : {SCRIPT_DIR}")
print()
