# Multi-Class Multimodal Gas Detection using ML

A full-stack gas detection system powered by a custom **SensorTransformer** — a Transformer Encoder that classifies hazardous gases from 7 MQ sensor readings in real time. Includes a professional monitoring dashboard, a FastAPI inference backend, and a React landing page.

---

## Live Demo

| Surface | URL |
|---------|-----|
| Landing Page | Deployed on Vercel |
| Live Dashboard | `/dashboard` route on the same Vercel deployment |
| API | Deployed on Render |

---

## What's Inside

### ML Models
Five models trained on 6,400 balanced sensor samples across 4 gas classes:

| Model | Accuracy |
|-------|----------|
| **SensorTransformer (Gen AI)** | **~98.5%** |
| K-Nearest Neighbors | 97.08% |
| Random Forest | 96.93% |
| Support Vector Machine | 91.00% |
| Logistic Regression | 83.00% |

### Gas Classes
- **NoGas** — baseline ambient air
- **Smoke** — combustion byproducts
- **Perfume** — alcohol vapors
- **Mixture** — combination of 2+ gases

### Sensor Inputs (7 MQ sensors)
| Sensor | Detects |
|--------|---------|
| MQ2 | Smoke, LPG, Methane |
| MQ3 | Alcohol / Perfume vapors |
| MQ5 | Natural Gas, LPG |
| MQ6 | LPG, Butane, Propane |
| MQ7 | Carbon Monoxide |
| MQ8 | Hydrogen |
| MQ135 | Air Quality — NH₃, CO₂, Benzene |

### SensorTransformer Architecture
```
7 sensor values
  → Linear projection (7 → 64)
  → Prepend [CLS] token + Positional Encoding
  → 2× TransformerEncoderLayer (4-head attention, d_ff=128, GELU, Pre-LN)
  → Extract [CLS] output
  → MLP head (64 → 32 → 4)
  → Softmax → predicted class + confidence
```

---

## Project Structure

```
├── backend/                        # FastAPI inference server
│   ├── main.py                     # /api/health, /api/random, /api/predict
│   ├── requirements.txt
│   ├── model.onnx                  # Exported SensorTransformer (no PyTorch needed)
│   ├── scaler.pkl                  # Fitted StandardScaler
│   ├── label_encoder.pkl           # Fitted LabelEncoder
│   └── sensor_pool.json            # Representative samples per class (for auto mode)
│
├── src/
│   ├── pages/
│   │   └── Dashboard.jsx           # /dashboard — live monitoring dashboard
│   ├── components/                 # Landing page sections
│   │   ├── Hero.jsx
│   │   ├── Dataset.jsx
│   │   ├── Models.jsx
│   │   ├── Transformer.jsx
│   │   ├── Results.jsx
│   │   └── ...
│   ├── App.jsx                     # Landing page (/ route)
│   └── main.jsx                    # BrowserRouter + route definitions
│
├── Multimodal Gas Detection & Classification Dataset/
│   └── Gas Sensors Measurements/
│       └── Gas_Sensors_Measurements.csv   # 6,400 samples, 4 classes
│
├── run_transformer.py              # Full training + evaluation script
├── export_model.py                 # Converts trained model → ONNX + pickles
├── demo_inference.py               # CLI demo: trains/loads model, runs dummy inputs
├── transformer_best.pt             # Saved PyTorch checkpoint
├── vercel.json                     # SPA rewrite rules for /dashboard route
├── vite.config.js                  # Vite + /api proxy for local dev
└── package.json
```

---

## Local Development

### 1. Train / run the ML model (Python)

```bash
# Install Python dependencies
pip install torch numpy pandas scikit-learn matplotlib seaborn

# Train the SensorTransformer and evaluate all models
python run_transformer.py

# Quick demo — runs inference on representative dummy sensor data
python demo_inference.py
```

### 2. Export model for the API (one time)

```bash
pip install onnx onnxruntime
python export_model.py
# Writes backend/model.onnx, backend/scaler.pkl, backend/label_encoder.pkl
```

### 3. Start the FastAPI backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
# API available at http://localhost:8001
# Docs at http://localhost:8001/docs
```

API endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Liveness check |
| `GET` | `/api/random` | Random sensor reading + model prediction |
| `POST` | `/api/predict` | Predict from user-supplied sensor values |

### 4. Start the React frontend

```bash
# In the project root (where package.json is)
npm install
npm run dev
# Opens at http://localhost:5173 (or next available port)
# /api/* requests are proxied to localhost:8001 automatically
```

- **`/`** — landing page
- **`/dashboard`** — live monitoring dashboard (auto feed + manual slider mode)

---

## Deployment

### Backend → Render (free tier)

1. Go to [render.com](https://render.com) → **New → Web Service**
2. Connect the GitHub repo
3. Set these fields:

   | Field | Value |
   |-------|-------|
   | Root Directory | `Multi-Class-Multimodal-Gas-Detection-using-ML/backend` |
   | Runtime | Python 3 |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
   | Instance Type | Free |

4. Click **Deploy**. Copy the URL (e.g. `https://gas-detection-api.onrender.com`)
5. Verify: open `https://<your-url>.onrender.com/api/health`

> **Note:** Render free instances sleep after 15 min of inactivity. The dashboard shows a `CONNECTING…` indicator and auto-retries on cold start (~30 s).

### Frontend → Vercel (free tier)

1. Go to [vercel.com](https://vercel.com) → **New Project → Import** the repo
2. Set these fields:

   | Field | Value |
   |-------|-------|
   | Root Directory | `Multi-Class-Multimodal-Gas-Detection-using-ML` |
   | Framework Preset | Vite |
   | Build Command | `npm run build` |
   | Output Directory | `dist` |

3. Add an **Environment Variable**:

   | Key | Value |
   |-----|-------|
   | `VITE_API_URL` | `https://<your-render-url>.onrender.com` |

4. Click **Deploy**. Your app goes live at `https://<project>.vercel.app`

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | PyTorch — custom SensorTransformer |
| Model Export | ONNX (via `torch.onnx.export`) |
| Inference Runtime | `onnxruntime` (no PyTorch needed in prod) |
| API | FastAPI + Uvicorn |
| Frontend | React 18 + Vite |
| Styling | Tailwind CSS |
| Animations | Framer Motion |
| Icons | Lucide React |
| Frontend Hosting | Vercel |
| Backend Hosting | Render |
