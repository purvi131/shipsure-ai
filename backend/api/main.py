from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import numpy as np
import os

# ─────────────────────────────────────────────
# LOAD MODEL + ENCODERS
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')

try:
    model         = joblib.load(os.path.join(MODEL_DIR, 'shipsure_model.pkl'))
    cat_encoder   = joblib.load(os.path.join(MODEL_DIR, 'category_encoder.pkl'))
    subcat_encoder= joblib.load(os.path.join(MODEL_DIR, 'subcategory_encoder.pkl'))
    feature_list  = joblib.load(os.path.join(MODEL_DIR, 'feature_list.pkl'))
    print("✅ Model loaded successfully")
except Exception as e:
    raise RuntimeError(f"Could not load model files: {e}")

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="ShipSure-AI API",
    description="Return risk detection for Indian e-commerce orders",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────

class OrderInput(BaseModel):
    order_value: float = Field(..., example=1499)
    category: Literal['Clothing', 'Electronics', 'Jewelry', 'Home & Kitchen']
    subcategory: str = Field(..., example="Women")
    payment_type: Literal['COD', 'Prepaid']
    past_returns: int = Field(..., ge=0, le=10, example=2)
    pincode: int = Field(..., example=411045)
    pincode_tier: Literal[1, 2, 3]
    days_since_account_creation: int = Field(..., ge=1, example=120)
    device_type: Literal['mobile', 'desktop']
    order_quantity: int = Field(..., ge=1, example=1)

class RiskResponse(BaseModel):
    risk_score: int           # 0–100
    risk_label: str           # Low / Medium / High
    recommendation: str       # action for the store
    return_probability: float # raw probability

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_risk_label(score: int) -> str:
    if score <= 40:
        return 'Low'
    elif score <= 70:
        return 'Medium'
    else:
        return 'High'

def get_recommendation(label: str, payment_type: str) -> str:
    if label == 'Low':
        return "Process order normally."
    elif label == 'Medium':
        if payment_type == 'COD':
            return "Send OTP verification to customer before confirming."
        else:
            return "Show order confirmation page to customer."
    else:  # High
        return "Place order under review. Notify customer within 2 hours."

def encode_features(order: OrderInput) -> list:
    # Encode category
    try:
        category_enc = cat_encoder.transform([order.category])[0]
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown category: {order.category}")

    # Encode subcategory
    try:
        subcategory_enc = subcat_encoder.transform([order.subcategory])[0]
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown subcategory: {order.subcategory}")

    payment_type_enc = 1 if order.payment_type == 'COD' else 0
    device_mobile    = 1 if order.device_type == 'mobile' else 0
    is_new_account   = 1 if order.days_since_account_creation < 30 else 0
    is_high_returner = 1 if order.past_returns >= 4 else 0
    is_bulk_order    = 1 if order.order_quantity >= 3 else 0
    value_per_item   = order.order_value / order.order_quantity

    feature_map = {
        'order_value':                  order.order_value,
        'category_enc':                 category_enc,
        'subcategory_enc':              subcategory_enc,
        'payment_type_enc':             payment_type_enc,
        'past_returns':                 order.past_returns,
        'pincode_tier':                 order.pincode_tier,
        'days_since_account_creation':  order.days_since_account_creation,
        'device_mobile':                device_mobile,
        'order_quantity':               order.order_quantity,
        'is_new_account':               is_new_account,
        'is_high_returner':             is_high_returner,
        'is_bulk_order':                is_bulk_order,
        'value_per_item':               value_per_item,
    }

    return [feature_map[f] for f in feature_list]

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "ShipSure-AI is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=RiskResponse)
def predict(order: OrderInput):
    # Build feature vector
    features = encode_features(order)
    features_array = np.array(features).reshape(1, -1)

    # Predict
    prob = float(model.predict_proba(features_array)[0][1])
    score = int(prob * 100)
    label = get_risk_label(score)
    recommendation = get_recommendation(label, order.payment_type)

    return RiskResponse(
        risk_score=score,
        risk_label=label,
        recommendation=recommendation,
        return_probability=round(prob, 4)
    )
